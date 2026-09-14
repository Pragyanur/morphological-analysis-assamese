"""
error_analysis.py  --  corrected candidate generation + validation analysis.

Summary of what changed relative to the original
------------------------------------------------

1.  `transition_log_probability` returned 0.0 == log(1.0) for any transition
    absent from the bigram matrix.  Every real transition contributes a
    negative number, so 0.0 was the *maximum* achievable score: sequences made
    entirely of unattested transitions were ranked FIRST.  Replaced by an
    interpolated (backoff) estimate that is never 0 and never 1.

2.  `get_option_tags("None of the above")` iterates a string, so it returned
    [] -> first_tag/last_tag None -> every transition 0.0.  The all-"None of
    the above" sequence was therefore the single best-scoring candidate in the
    pool.  Such options now get an explicit `<none>` tag and a configurable
    penalty.

3.  Scores summed one term per *morpheme transition*, and candidates differ in
    morpheme count, so shorter analyses were systematically cheaper -- exactly
    backwards for an agglutinative language.  Fixed by modelling an explicit
    end-of-word event, so an analysis pays for stopping where it stops.
    NOTE: this only removes the bias if P(EOW | tag) is properly estimated;
    see `TransitionModel.from_corpus`.  With a conditional-probability CSV
    alone the end-of-word term is constant and cancels (see docstring).

4.  The "group by last tag before pruning" block was a no-op: the global
    `nlargest(k)` two statements later discarded states by score with no
    regard to tag.  Replaced with a real k-best DP that keeps a k-best list
    *per last tag* (states merge under the Markov property), plus an optional
    reserve-then-fill cap that actually delivers the promised diversity.

5.  Evaluation counted a word as correct when gold was not found in the option
    list, because both indices were -1 and `-1 == -1`.  Now a missing gold is
    always incorrect, and is reported separately.

6.  Assorted: O(n^2) pandas `.loc` loading, duplicated batching code between
    the exhaustive and top-k branches, `evaluate_top_candidates` taking an
    unused argument, the padding mask inferred from all-zero rows (which
    masks a genuine all-zero feature vector), `annotated_list` indexed without
    a length guard, `total_candidate_analyses` collapsing to 0 when a word has
    no valid option, and the HTML report reading a "sentence" key that was
    never stored.

7.  Added the pruning diagnostics: whether the gold sequence survives into the
    top-k pool and at what rank.  That is the oracle recall of the heuristic
    and the ceiling on everything downstream.
"""

from __future__ import annotations

import heapq
import html as _html
import itertools
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================
# Configuration
# ============================================================

TOP_K = 250_000

BOS = "<bos>"          # sentence start context
EOS = "<eos>"          # sentence end symbol
EOW = "<eow>"          # end-of-word event, emitted after a word's last morpheme
NONE_TAG = "<none>"    # tag given to the "None of the above" pseudo-option

NONE_OPTION = "None of the above"

_LOG_FLOOR = 1e-300


# ============================================================
# Extract tags from one morphological option
# ============================================================

def get_option_tags(option: Any) -> List[str]:
    """
    Option format:

        [['morpheme_1', 'tag_1'], ['morpheme_2', 'tag_2'], ...]

    Returns ['tag_1', 'tag_2', ...].

    A non-structured option (notably the string "None of the above") yields
    [NONE_TAG] rather than [] -- an empty tag list is what allowed those
    options to score 0.0, the maximum, in the original code.
    """
    if option is None:
        return [NONE_TAG]

    if isinstance(option, str):
        return [NONE_TAG]

    tags: List[str] = []
    for morpheme_tag in option:
        if isinstance(morpheme_tag, (list, tuple)) and len(morpheme_tag) >= 2:
            tags.append(morpheme_tag[1])

    return tags if tags else [NONE_TAG]


def is_none_option(option: Any) -> bool:
    """True for the 'no valid analysis' pseudo-option."""
    return isinstance(option, str) and option.strip() == NONE_OPTION


# ============================================================
# Transition model
# ============================================================

class TransitionModel:
    """
    A properly normalised first-order model over morpheme-tag sequences.

    Three distributions, all smoothed by interpolation with a unigram so that
    an unseen event is unlikely but never impossible (and never certain):

        intra[t][t']   P(next morpheme tag t' | previous morpheme tag t)
                       within one word; the symbol EOW means "the word ends
                       here", which is what makes analyses of different
                       morpheme counts comparable.

        cross[t][t']   P(first tag of the next word t' | last tag of this
                       word t).  Context BOS is the sentence start.  This
                       keeps the cross-word dependency the original code had.

        eos[t]         P(sentence ends | last tag of last word t).

    Interpolation:   P~(x | c) = lam * P(x | c) + (1 - lam) * P_uni(x)

    so log P~ is always finite, always strictly negative, and an unattested
    transition is penalised rather than rewarded.
    """

    def __init__(
        self,
        intra: Dict[str, Dict[str, float]],
        cross: Dict[str, Dict[str, float]],
        eos: Dict[str, float],
        unigram_intra: Dict[str, float],
        unigram_cross: Dict[str, float],
        lam: float = 0.9,
        none_penalty: float = -10.0,
    ):
        self.intra = intra
        self.cross = cross
        self.eos = eos
        self.unigram_intra = unigram_intra
        self.unigram_cross = unigram_cross
        self.lam = float(lam)
        self.none_penalty = float(none_penalty)

        self._uniform_intra = 1.0 / max(len(unigram_intra), 1)
        self._uniform_cross = 1.0 / max(len(unigram_cross), 1)

        self._cache: Dict[Tuple[str, str, str], float] = {}

    # --------------------------------------------------------

    def _interp(
        self,
        table: Dict[str, Dict[str, float]],
        context: str,
        symbol: str,
        unigram: Dict[str, float],
        uniform: float,
    ) -> float:
        p_bi = table.get(context, {}).get(symbol, 0.0)
        p_uni = unigram.get(symbol, uniform)
        p = self.lam * p_bi + (1.0 - self.lam) * p_uni
        return math.log(max(p, _LOG_FLOOR))

    # --------------------------------------------------------

    def logp_intra(self, previous_tag: str, current_tag: str) -> float:
        """log P(current_tag | previous_tag), within a word."""
        key = ("i", previous_tag, current_tag)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._interp(
                self.intra, previous_tag, current_tag,
                self.unigram_intra, self._uniform_intra,
            )
            self._cache[key] = hit
        return hit

    def logp_eow(self, last_tag: str) -> float:
        """log P(word ends | last_tag)."""
        return self.logp_intra(last_tag, EOW)

    def logp_cross(self, previous_last_tag: str, first_tag: str) -> float:
        """log P(first tag of this word | last tag of the previous word)."""
        key = ("x", previous_last_tag, first_tag)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._interp(
                self.cross, previous_last_tag, first_tag,
                self.unigram_cross, self._uniform_cross,
            )
            self._cache[key] = hit
        return hit

    def logp_eos(self, last_tag: str) -> float:
        """log P(sentence ends | last tag of the final word)."""
        p = self.eos.get(last_tag)
        if p is None:
            # Not estimated (CSV-only mode): constant across candidates for a
            # fixed sentence, so it cancels out of the ranking.
            return 0.0
        return math.log(max(self.lam * p + (1.0 - self.lam) * 0.5, _LOG_FLOOR))

    # --------------------------------------------------------
    # Constructors
    # --------------------------------------------------------

    @classmethod
    def from_probability_csv(
        cls,
        bigram_file: str,
        lam: float = 0.9,
        none_penalty: float = -10.0,
    ) -> "TransitionModel":
        """
        Load the existing `empirical-bigram-probabilities.csv` (rows =
        previous tag, columns = next tag, cells = conditional probability).

        LIMITATION.  That matrix has no end-of-word symbol, so P(EOW | tag)
        falls back to the unigram term alone, which is the same value for
        every tag.  A constant per word cancels from the ranking (the number
        of words in a sentence is fixed), so this constructor fixes the
        inverted ranking and the "None of the above" hole, but NOT the bias
        toward morphologically shorter analyses.  To fix that too, re-estimate
        with `TransitionModel.from_corpus`, which counts end-of-word events
        from your training analyses.
        """
        df = pd.read_csv(bigram_file, index_col=0).astype(float)

        if df.index.duplicated().any():
            df = df.groupby(level=0).sum()

        # Row-normalise defensively: the file is meant to hold conditional
        # probabilities, but nothing guarantees the rows sum to 1.
        row_sums = df.sum(axis=1)
        nonzero = row_sums > 0
        df.loc[nonzero] = df.loc[nonzero].div(row_sums[nonzero], axis=0)

        table: Dict[str, Dict[str, float]] = {}
        for previous_tag, row in df.to_dict(orient="index").items():
            table[str(previous_tag)] = {
                str(next_tag): float(p)
                for next_tag, p in row.items()
                if p and p > 0.0
            }

        # No marginal counts are recoverable from conditional rows alone, so
        # approximate the unigram by averaging the rows.
        column_totals: Dict[str, float] = defaultdict(float)
        for row in table.values():
            for symbol, p in row.items():
                column_totals[symbol] += p
        total = sum(column_totals.values()) or 1.0
        unigram = {s: v / total for s, v in column_totals.items()}

        return cls(
            intra=table,
            cross=table,
            eos={},
            unigram_intra=unigram,
            unigram_cross=unigram,
            lam=lam,
            none_penalty=none_penalty,
        )

    @classmethod
    def from_corpus(
        cls,
        sentences: Iterable[Sequence[Any]],
        lam: float = 0.9,
        alpha: float = 0.1,
        none_penalty: float = -10.0,
    ) -> "TransitionModel":
        """
        Estimate every distribution from gold analyses -- the recommended path.

        `sentences` is an iterable of sentences, each a sequence of gold
        options in the [[morpheme, tag], ...] format (i.e. exactly the
        `group["annotated_list"]` you already have for the training split).

        `alpha` is add-alpha smoothing applied before the interpolation.
        """
        intra_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        cross_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        eos_counts: Dict[str, float] = defaultdict(float)
        vocab = {EOW, NONE_TAG}

        n_sentences = 0
        for sentence in sentences:
            n_sentences += 1
            previous_last = BOS
            last_tag = None

            for option in sentence:
                tags = get_option_tags(option)
                vocab.update(tags)

                cross_counts[previous_last][tags[0]] += 1.0
                for j in range(1, len(tags)):
                    intra_counts[tags[j - 1]][tags[j]] += 1.0
                intra_counts[tags[-1]][EOW] += 1.0

                previous_last = tags[-1]
                last_tag = tags[-1]

            if last_tag is not None:
                eos_counts[last_tag] += 1.0

        if n_sentences == 0:
            raise ValueError("from_corpus received no sentences")

        intra_symbols = sorted(vocab | {EOW})
        cross_symbols = sorted(vocab)

        intra = _normalise_counts(intra_counts, intra_symbols, alpha)
        cross = _normalise_counts(cross_counts, cross_symbols, alpha)

        unigram_intra = _marginal(intra_counts, intra_symbols, alpha)
        unigram_cross = _marginal(cross_counts, cross_symbols, alpha)

        # P(sentence ends | last tag of the word) = (times that tag ended the
        # sentence) / (times that tag ended any word).
        eos: Dict[str, float] = {}
        for tag in vocab:
            words_ending_with_tag = intra_counts.get(tag, {}).get(EOW, 0.0)
            sentences_ending_with_tag = eos_counts.get(tag, 0.0)
            denominator = max(words_ending_with_tag, sentences_ending_with_tag)
            eos[tag] = (sentences_ending_with_tag + alpha) / (denominator + 2.0 * alpha)

        return cls(
            intra=intra,
            cross=cross,
            eos=eos,
            unigram_intra=unigram_intra,
            unigram_cross=unigram_cross,
            lam=lam,
            none_penalty=none_penalty,
        )


def _normalise_counts(
    counts: Dict[str, Dict[str, float]],
    symbols: Sequence[str],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    v = len(symbols)
    for context, row in counts.items():
        denominator = sum(row.values()) + alpha * v
        out[context] = {s: (row.get(s, 0.0) + alpha) / denominator for s in symbols}
    return out


def _marginal(
    counts: Dict[str, Dict[str, float]],
    symbols: Sequence[str],
    alpha: float,
) -> Dict[str, float]:
    totals: Dict[str, float] = {s: alpha for s in symbols}
    for row in counts.values():
        for s, c in row.items():
            totals[s] = totals.get(s, alpha) + c
    grand = sum(totals.values()) or 1.0
    return {s: c / grand for s, c in totals.items()}


def load_bigram_probabilities(bigram_file: str, lam: float = 0.9) -> TransitionModel:
    """
    Backwards-compatible entry point.  Returns a TransitionModel now, not a
    raw nested dict -- the raw dict is what let missing transitions be scored
    as certainties.
    """
    return TransitionModel.from_probability_csv(bigram_file, lam=lam)


# ============================================================
# Precompute information for every morphological option
# ============================================================

def prepare_options(
    sentence_options: Sequence[Sequence[Any]],
    transition_model: TransitionModel,
) -> List[List[Dict[str, Any]]]:
    """
    Per candidate analysis, precompute the context-independent part of its
    score:

        internal_score = sum_j log P(t_j | t_{j-1})  +  log P(EOW | t_last)

    The EOW term is what makes analyses with different morpheme counts
    comparable; without it, fewer morphemes always meant a higher score.
    """
    tm = transition_model
    prepared: List[List[Dict[str, Any]]] = []

    for word_options in sentence_options:
        word_prepared: List[Dict[str, Any]] = []

        for option in word_options:
            tags = get_option_tags(option)

            internal = 0.0
            for j in range(1, len(tags)):
                internal += tm.logp_intra(tags[j - 1], tags[j])
            internal += tm.logp_eow(tags[-1])

            if is_none_option(option):
                # An explicit, tunable cost for "the analyser found nothing",
                # instead of the free ride it used to get.
                internal += tm.none_penalty

            word_prepared.append(
                {
                    "option": option,
                    "tags": tags,
                    "first_tag": tags[0],
                    "last_tag": tags[-1],
                    "internal_score": internal,
                }
            )

        prepared.append(word_prepared)

    return prepared


# ============================================================
# TOP-K dynamic programming (k-best over the lattice)
# ============================================================

def _extend(entries, delta: float, option_index: int):
    """
    Lazily add a constant to a descending-sorted list of (score, indices).
    The result stays sorted, which is what lets heapq.merge do the k-best
    merge without re-sorting.

    This must be a function, not an inline generator expression: a generator
    expression would capture `delta` and `option_index` by reference and read
    them after the enclosing loop had moved on.
    """
    return ((score + delta, indices + (option_index,)) for score, indices in entries)


def _cap_states(
    buckets: Dict[str, List[Tuple[float, tuple]]],
    max_states: int,
    reserve_per_tag: int,
) -> Dict[str, List[Tuple[float, tuple]]]:
    """
    Bound total memory while genuinely preserving last-tag diversity: every
    bucket keeps `reserve_per_tag` entries unconditionally, then the remaining
    budget is filled by global score.  (The original code claimed to do this
    but the global nlargest afterwards threw the guarantee away.)
    """
    total = sum(len(v) for v in buckets.values())
    if total <= max_states:
        return buckets

    n_buckets = max(len(buckets), 1)
    reserve = max(0, min(reserve_per_tag, max_states // n_buckets))

    kept: Dict[str, List[Tuple[float, tuple]]] = {}
    rest: List[Tuple[str, Tuple[float, tuple]]] = []

    for tag, entries in buckets.items():
        kept[tag] = list(entries[:reserve])
        for entry in entries[reserve:]:
            rest.append((tag, entry))

    remaining = max_states - sum(len(v) for v in kept.values())
    if remaining > 0 and rest:
        for tag, entry in heapq.nlargest(remaining, rest, key=lambda te: te[1][0]):
            kept[tag].append(entry)

    for tag in kept:
        kept[tag].sort(key=lambda e: e[0], reverse=True)

    return {tag: entries for tag, entries in kept.items() if entries}


def get_top_probability_candidates(
    sentence_options: Sequence[Sequence[Any]],
    transition_model: TransitionModel,
    k: int = TOP_K,
    max_states: Optional[int] = None,
    reserve_per_tag: int = 1,
) -> List[Tuple[float, tuple]]:
    """
    Exact k-best sequences under the transition model, by dynamic programming
    over the lattice.

    State = last morpheme tag of the prefix.  Under a first-order model that
    tag is a sufficient statistic for the future, so two prefixes sharing it
    are interchangeable and can be merged into one k-best list.  The original
    code kept every prefix separately, which is beam search: more memory, and
    approximate.

    Cost is O(n * |tags| * options * k) time and O(|tags| * k) states, against
    O(k * options) states before.

    Returns [(score, option_index_tuple), ...], best first.
    """
    tm = transition_model
    prepared = prepare_options(sentence_options, tm)

    if max_states is None:
        max_states = k

    buckets: Dict[str, List[Tuple[float, tuple]]] = {BOS: [(0.0, tuple())]}

    for word_index, word_options in enumerate(prepared):
        if not word_options:
            raise ValueError(f"word {word_index} has no candidate analyses")

        streams: Dict[str, List[Iterator]] = defaultdict(list)

        for previous_tag, entries in buckets.items():
            if not entries:
                continue
            for option_index, option in enumerate(word_options):
                delta = (
                    tm.logp_cross(previous_tag, option["first_tag"])
                    + option["internal_score"]
                )
                streams[option["last_tag"]].append(
                    _extend(entries, delta, option_index)
                )

        new_buckets: Dict[str, List[Tuple[float, tuple]]] = {}
        for last_tag, iterators in streams.items():
            merged = heapq.merge(*iterators, key=lambda e: e[0], reverse=True)
            new_buckets[last_tag] = list(itertools.islice(merged, k))

        buckets = _cap_states(new_buckets, max_states, reserve_per_tag)

    finals = [
        _extend(entries, tm.logp_eos(last_tag), -1)
        for last_tag, entries in buckets.items()
    ]
    merged = heapq.merge(*finals, key=lambda e: e[0], reverse=True)

    # _extend appended a sentinel -1 for the EOS step; strip it.
    return [
        (score, indices[:-1])
        for score, indices in itertools.islice(merged, k)
    ]


def score_index_sequence(
    sentence_options: Sequence[Sequence[Any]],
    indices: Sequence[int],
    transition_model: TransitionModel,
) -> float:
    """
    Score one explicit choice of option indices under the same model -- used
    to place the gold sequence on the same scale as the candidate pool.
    """
    tm = transition_model
    prepared = prepare_options(sentence_options, transition_model)

    score = 0.0
    previous_tag = BOS
    for word_index, option_index in enumerate(indices):
        option = prepared[word_index][option_index]
        score += tm.logp_cross(previous_tag, option["first_tag"])
        score += option["internal_score"]
        previous_tag = option["last_tag"]

    return score + tm.logp_eos(previous_tag)


# ============================================================
# Convert top-K index sequences into model candidates
# ============================================================

def materialize_top_candidates(
    top_candidates: Iterable[Tuple[float, tuple]],
    vectorized_options: Sequence[Sequence[Tuple[np.ndarray, Any]]],
) -> Iterator[Tuple[float, tuple]]:
    """
    Convert (score, option_index_sequence) into (score, combo), where combo
    matches the structure of itertools.product(*vectorized_options).

    A generator now: materialising 250k tuples-of-tuples up front was a large
    and needless memory spike, and the consumer iterates exactly once.
    """
    for probability_score, indices in top_candidates:
        combo = tuple(
            vectorized_options[word_index][option_index]
            for word_index, option_index in enumerate(indices)
        )
        yield probability_score, combo


# ============================================================
# Transformer scoring
# ============================================================

def _score_batch(
    chunk_vectors: List[torch.Tensor],
    chunk_lengths: List[int],
    model,
    device,
) -> torch.Tensor:
    input_tensor = torch.stack(chunk_vectors).to(device, non_blocking=True)

    # Mask from true lengths, not from all-zero rows: a genuine feature
    # vector that happens to be all zeros (the "None of the above" vector,
    # for instance) would otherwise be masked out as padding.
    window = input_tensor.shape[1]
    positions = torch.arange(window, device=device).unsqueeze(0)
    lengths = torch.tensor(chunk_lengths, device=device).unsqueeze(1)
    mask = positions >= lengths

    with torch.no_grad():
        logits = model(input_tensor, mask=mask)
        logits = logits.reshape(logits.shape[0], -1)[:, 0]

    return logits


def rank_candidates(
    candidates: Iterable[Tuple[Any, ...]],
    model,
    device,
    window_size: int,
    feature_dim: int,
    eval_batch_size: int,
) -> Tuple[float, Optional[list], int, int]:
    """
    Score an iterable of candidate combos and return the best-scoring one.

    `candidates` yields combos of (vector, option) pairs -- the structure
    produced both by itertools.product(*vectorized_options) and by
    materialize_top_candidates (whose score element is stripped by the
    caller).

    Returns (best_score, best_labels, n_scored, n_truncated).
    """
    padding_vec = np.zeros(feature_dim, dtype=np.float32)

    current_vectors: List[torch.Tensor] = []
    current_lengths: List[int] = []
    current_labels: List[list] = []

    best_score = -float("inf")
    best_labels: Optional[list] = None
    n_scored = 0
    n_truncated = 0

    def flush():
        nonlocal best_score, best_labels, current_vectors, current_lengths, current_labels
        if not current_vectors:
            return
        logits = _score_batch(current_vectors, current_lengths, model, device)
        best_idx = int(torch.argmax(logits).item())
        chunk_best = float(logits[best_idx].item())
        if chunk_best > best_score:
            best_score = chunk_best
            best_labels = current_labels[best_idx]
        current_vectors = []
        current_lengths = []
        current_labels = []

    for combo in candidates:
        seq_vectors = [item[0] for item in combo]
        seq_labels = [item[1] for item in combo]

        if len(seq_vectors) > window_size:
            n_truncated += 1
            seq_vectors = seq_vectors[:window_size]
            seq_labels = seq_labels[:window_size]

        length = len(seq_vectors)
        padded = list(seq_vectors) + [padding_vec] * (window_size - length)

        current_vectors.append(
            torch.from_numpy(np.asarray(padded, dtype=np.float32))
        )
        current_lengths.append(length)
        current_labels.append(seq_labels)
        n_scored += 1

        if len(current_vectors) >= eval_batch_size:
            flush()

    flush()

    return best_score, best_labels, n_scored, n_truncated


# ============================================================
# Gold-sequence diagnostics
# ============================================================

def gold_index_tuple(
    sentence_options: Sequence[Sequence[Any]],
    annotated_list: Sequence[Any],
) -> Tuple[List[int], bool]:
    """
    Map the gold analyses onto option indices.  Returns (indices, complete),
    where `complete` is False if any gold analysis is absent from the
    analyser's candidate list -- that word is unrecoverable by any model, and
    must not be scored as correct.
    """
    indices: List[int] = []
    complete = True

    for i, word_options in enumerate(sentence_options):
        if i >= len(annotated_list):
            indices.append(-1)
            complete = False
            continue
        try:
            indices.append(list(word_options).index(annotated_list[i]))
        except ValueError:
            indices.append(-1)
            complete = False

    return indices, complete


def find_gold_rank(
    top_candidates: Sequence[Tuple[float, tuple]],
    gold_indices: Sequence[int],
) -> Optional[int]:
    """1-based rank of the gold sequence in the pruned pool, or None."""
    if any(i < 0 for i in gold_indices):
        return None
    target = tuple(gold_indices)
    for rank, (_, indices) in enumerate(top_candidates, start=1):
        if indices == target:
            return rank
    return None


# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def validation_analysis(
    val_groups,
    model,
    device,
    morphology_generator,
    vectorize_func,
    window_size,
    feature_dim,
    eval_batch_size=16384,
    bigram_file="resources/empirical-bigram-probabilities.csv",
    max_probability_candidates=TOP_K,
    candidate_stats_file="sentence_error_analysis.csv",
    transition_model: Optional[TransitionModel] = None,
    lam: float = 0.9,
    none_penalty: float = -10.0,
    max_states: Optional[int] = None,
    reserve_per_tag: int = 1,
):
    """
    `transition_model` overrides `bigram_file`.  Build one with
    TransitionModel.from_corpus(training_analyses) to also remove the bias
    toward morphologically shorter analyses -- see that method's docstring.
    """
    model.eval()

    sentence_details = []
    candidate_statistics = []

    if transition_model is None:
        print("Loading empirical bigram probabilities...")
        transition_model = TransitionModel.from_probability_csv(
            bigram_file, lam=lam, none_penalty=none_penalty
        )
        print(
            f"Loaded {len(transition_model.intra):,} previous-tag entries. "
            "(No end-of-word statistics in this file: the length bias is not "
            "corrected. Use TransitionModel.from_corpus to fix it.)"
        )

    for group_number, group in enumerate(val_groups, start=1):

        sent_text = group["sentence"]
        annotated_list = group.get("annotated_list", [])

        print("\n" + "=" * 70)
        print(f"Sentence {group_number}/{len(val_groups)}")
        print(sent_text)

        sentence_options = morphology_generator(sent_text)

        # ----------------------------------------------------
        # Sentence-level statistics
        # ----------------------------------------------------

        sentence_length = len(sentence_options)

        valid_counts = [
            sum(1 for opt in word_options if not is_none_option(opt))
            for word_options in sentence_options
        ]
        full_counts = [len(word_options) for word_options in sentence_options]

        ambiguous_words = sum(1 for c in valid_counts if c > 1)

        # Product over the counts actually used for the branch decision;
        # the old version used the filtered counts, so a word whose only
        # option was "None of the above" collapsed the product to zero.
        total_candidate_analyses = 1
        for count in full_counts:
            total_candidate_analyses *= max(count, 1)

        # ----------------------------------------------------
        # Vectorize
        # ----------------------------------------------------

        vectorized_options = [
            [(vectorize_func(opt), opt) for opt in word_options]
            for word_options in sentence_options
        ]

        total_candidates = 1
        for options in vectorized_options:
            total_candidates *= len(options)

        print(f"Total possible sequences: {total_candidates:,}")

        gold_idx, gold_complete = gold_index_tuple(sentence_options, annotated_list)
        gold_rank: Optional[int] = None
        gold_in_pool: Optional[bool] = None

        # ----------------------------------------------------
        # CASE 1: exhaustive
        # ----------------------------------------------------

        if total_candidates <= max_probability_candidates:
            print("Inference mode: EXHAUSTIVE")
            inference_mode = "exhaustive"

            best_score, best_combo_labels, n_scored, n_truncated = rank_candidates(
                itertools.product(*vectorized_options),
                model,
                device,
                window_size,
                feature_dim,
                eval_batch_size,
            )

            if gold_complete:
                gold_in_pool = True
                gold_rank = None  # every sequence is in the pool

        # ----------------------------------------------------
        # CASE 2: top-k pruning
        # ----------------------------------------------------

        else:
            print("Inference mode: TOP-K BIGRAM PROBABILITY")
            inference_mode = "top_k"
            print(
                f"Generating top {max_probability_candidates:,} "
                "probability candidates..."
            )

            top_candidates = get_top_probability_candidates(
                sentence_options,
                transition_model,
                k=max_probability_candidates,
                max_states=max_states,
                reserve_per_tag=reserve_per_tag,
            )

            print(f"Generated {len(top_candidates):,} candidates.")

            if gold_complete:
                gold_rank = find_gold_rank(top_candidates, gold_idx)
                gold_in_pool = gold_rank is not None
                if not gold_in_pool:
                    gold_score = score_index_sequence(
                        sentence_options, gold_idx, transition_model
                    )
                    print(
                        "  GOLD PRUNED AWAY  "
                        f"(gold score {gold_score:.3f}, "
                        f"worst kept {top_candidates[-1][0]:.3f})"
                    )

            print("Running Transformer inference...")

            best_score, best_combo_labels, n_scored, n_truncated = rank_candidates(
                (combo for _, combo in materialize_top_candidates(
                    top_candidates, vectorized_options
                )),
                model,
                device,
                window_size,
                feature_dim,
                eval_batch_size,
            )

        if n_truncated:
            print(
                f"  WARNING: {n_truncated:,} candidates were truncated to "
                f"window_size={window_size}; the tail of this sentence cannot "
                "be predicted."
            )

        # ----------------------------------------------------
        # Predicted indices
        # ----------------------------------------------------

        predicted_idx: List[int] = []

        for i, word_options in enumerate(sentence_options):
            if best_combo_labels is None or i >= len(best_combo_labels):
                predicted_idx.append(-1)
                if best_combo_labels is not None:
                    print(f"PREDICTED LENGTH MISMATCH at word {i}: {sent_text}")
                continue

            try:
                predicted_idx.append(
                    list(word_options).index(best_combo_labels[i])
                )
            except ValueError:
                predicted_idx.append(-1)
                print(f"PREDICTED MISMATCH at word {i}: {sent_text}")

        for i, index in enumerate(gold_idx):
            if index < 0:
                print(
                    f"ANNOTATED MISMATCH at word {i}: gold analysis is not "
                    f"among the analyser's options -- {sent_text}"
                )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        sentence_details.append(
            {
                "sentence": sent_text,
                "sentence-id": group.get("sentence-id", group_number),
                "words-options": sentence_options,
                "annotated-indices": gold_idx,
                "predicted-indices": predicted_idx,
            }
        )

        # A word counts as correct only when gold was actually findable.
        # Previously gold = -1 and prediction = -1 compared equal and scored
        # as a hit.
        correct_words = sum(
            1
            for g, p in zip(gold_idx, predicted_idx)
            if g >= 0 and g == p
        )
        unreachable_words = sum(1 for g in gold_idx if g < 0)
        incorrect_words = sentence_length - correct_words

        candidate_statistics.append(
            {
                "sentence": sent_text,
                "length": sentence_length,
                "ambiguous_words": ambiguous_words,
                "incorrect_words": incorrect_words,
                "unreachable_gold_words": unreachable_words,
                "candidate_analyses": total_candidate_analyses,
                "inference_mode": inference_mode,
                "gold_in_candidate_pool": gold_in_pool,
                "gold_rank_in_pool": gold_rank,
                "best_model_score": best_score,
            }
        )

    stats_df = pd.DataFrame(candidate_statistics)
    stats_df.to_csv(candidate_stats_file, index=False, encoding="utf-8")

    print(f"\nSaved sentence-level error analysis to: {candidate_stats_file}")

    pooled = stats_df[stats_df["inference_mode"] == "top_k"]
    if len(pooled):
        recall = pooled["gold_in_candidate_pool"].fillna(False).mean()
        print(
            f"Oracle recall of the top-k pool: {recall:.4f} "
            f"over {len(pooled):,} pruned sentences "
            "(this is the ceiling on the model's accuracy there)."
        )

    return sentence_details


# ============================================================
# HTML report
# ============================================================

def generate_sentence_html(record, index):
    sent_id = record.get("sentence-id", f"Batch Index: {index}")
    sentence_text = record.get("sentence", "")

    words_options = record["words-options"]
    annotated_indices = record["annotated-indices"]
    predicted_indices = record["predicted-indices"]

    html = f"""
    <div class="card">
        <div class="card-header">Sentence ID: {_html.escape(str(sent_id))}</div>
        <div class="card-body">
            {f'<p class="sentence-text">"{_html.escape(str(sentence_text))}"</p>' if sentence_text else ''}
            <table>
                <thead>
                    <tr>
                        <th style="width: 15%;">Word Position</th>
                        <th>Options Sequence</th>
                    </tr>
                </thead>
                <tbody>
    """

    for i, options in enumerate(words_options):
        gold = int(annotated_indices[i])
        pred = int(predicted_indices[i])

        opt_html_list = []
        for idx, opt in enumerate(options):
            if idx == gold and idx == pred:
                style_class = "badge badge-coincide"
            elif idx == gold:
                style_class = "badge badge-annotated"
            elif idx == pred:
                style_class = "badge badge-predicted"
            else:
                style_class = "badge-none"

            opt_html_list.append(
                f'<span class="{style_class}">{_html.escape(str(opt))}</span>'
            )

        if gold < 0:
            opt_html_list.append(
                '<span class="badge badge-missing">gold not in options</span>'
            )

        options_inline = " <span class='divider'>|</span> ".join(opt_html_list)
        html += f"""
                    <tr>
                        <td class="pos-cell">Word {i + 1}</td>
                        <td>{options_inline}</td>
                    </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return html


def build_combined_html(
    error_cards,
    correct_cards,
    total_error_words,
    total_correct_words,
    total_exact_correct_words,
    total_ambiguous_words,
    correct_ambiguous_words,
    total_unreachable_words,
    title,
):
    css_styles = """
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }
        h2 { color: #2c3e50; margin-bottom: 5px; }
        .section-heading { color: #1e293b; padding-bottom: 10px; border-bottom: 2px solid #cbd5e1; margin-top: 40px; margin-bottom: 20px; }
        .error-title { color: #dc3545; border-color: #f5c6cb; }
        .correct-title { color: #28a745; border-color: #c3e6cb; }
        .dashboard { display: flex; gap: 20px; margin-bottom: 25px; flex-wrap: wrap; }
        .panel { background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; flex: 1; min-width: 250px; }
        .metric-value { font-size: 1.15em; font-weight: bold; color: #0f172a; margin-top: 5px; line-height: 1.5; }
        .legend-item { display: inline-block; margin-right: 25px; font-weight: bold; font-size: 0.9em; }
        .nav-links a { display: inline-block; margin-right: 15px; color: #007bff; font-weight: bold; text-decoration: none; }
        .nav-links a:hover { text-decoration: underline; }
        .card { background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; border: 1px solid #e2e8f0; overflow: hidden; }
        .card-header { background-color: #f1f5f9; padding: 12px 20px; font-weight: bold; color: #475569; font-size: 0.9em; border-bottom: 1px solid #e2e8f0; }
        .card-body { padding: 20px; }
        .sentence-text { font-style: italic; font-size: 1.1em; color: #1e293b; margin-top: 0; margin-bottom: 15px; }
        table { width: 100%; border-collapse: collapse; }
        th { background-color: #f8fafc; text-align: left; padding: 10px; font-size: 0.85em; color: #64748b; border-bottom: 2px solid #cbd5e1; text-transform: uppercase; }
        td { padding: 10px; border-bottom: 1px solid #f1f5f9; font-size: 0.95em; }
        .pos-cell { font-weight: bold; color: #64748b; }
        .divider { color: #cbd5e1; margin: 0 6px; }
        .badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; }
        .badge-coincide { background-color: #28a745; color: white; }
        .badge-annotated { background-color: #007bff; color: white; }
        .badge-predicted { background-color: #ffc107; color: black; }
        .badge-missing { background-color: #6b7280; color: white; }
        .badge-none { color: #333333; padding: 4px 8px; }
    </style>
    """

    total_sentences = len(error_cards) + len(correct_cards)
    total_words = total_error_words + total_correct_words

    sentence_accuracy = (len(correct_cards) / total_sentences * 100) if total_sentences else 0.0
    word_accuracy = (total_exact_correct_words / total_words * 100) if total_words else 0.0
    ambiguous_accuracy = (
        correct_ambiguous_words / total_ambiguous_words * 100
    ) if total_ambiguous_words else 0.0

    reachable_words = total_words - total_unreachable_words
    reachable_accuracy = (
        total_exact_correct_words / reachable_words * 100
    ) if reachable_words else 0.0

    summary_panel = f"""
    <div class="dashboard">
        <div class="panel">
            <strong>Sentence-Level Metrics:</strong>
            <div class="metric-value">
                Total Sentences: {total_sentences}<br>
                <span style="color: #28a745;">Correct: {len(correct_cards)}</span> |
                <span style="color: #dc3545;">Errors: {len(error_cards)}</span>
                <br><small style="color: #64748b; font-weight: normal;">Sentence Accuracy: {sentence_accuracy:.2f}%</small>
            </div>
        </div>
        <div class="panel">
            <strong>Overall Word-Level Performance:</strong>
            <div class="metric-value">
                Total Words: {total_words}<br>
                <span style="color: #28a745;">Correct Predictions: {total_exact_correct_words}</span>
                <br><span style="color: #007bff;">Overall Word Accuracy: {word_accuracy:.2f}%</span>
                <br><small style="color: #64748b; font-weight: normal;">Gold unreachable (analyser miss): {total_unreachable_words} &middot; accuracy on reachable words: {reachable_accuracy:.2f}%</small>
            </div>
        </div>
        <div class="panel" style="border-left: 4px solid #ffc107;">
            <strong>Ambiguous Words Benchmark:</strong>
            <div class="metric-value">
                Total Ambiguous Words: {total_ambiguous_words}<br>
                <span style="color: #28a745;">Correctly Disambiguated: {correct_ambiguous_words}</span>
                <br><span style="color: #d97706;">Ambiguous Word Accuracy: {ambiguous_accuracy:.2f}%</span>
            </div>
        </div>
    </div>
    <div class="dashboard" style="margin-top: -10px;">
        <div class="panel">
            <strong>Legend:</strong>
            <div class="legend-item" style="margin-left: 15px;"><span class="badge badge-annotated">Blue</span> Annotated Index</div>
            <div class="legend-item"><span class="badge badge-predicted">Yellow</span> Predicted Index</div>
            <div class="legend-item"><span class="badge badge-coincide">Green</span> Coincide (Correct Word)</div>
            <div class="legend-item"><span class="badge badge-missing">Grey</span> Gold missing from options</div>
        </div>
        <div class="panel nav-links">
            <strong>Jump To:</strong>
            <a href="#errors-section" style="margin-left: 15px;">Incorrect Sentences ({len(error_cards)})</a>
            <a href="#correct-section">Perfect Sentences ({len(correct_cards)})</a>
        </div>
    </div>
    """

    errors_content = "".join(error_cards) if error_cards else "<p>No analytical errors found!</p>"
    correct_content = "".join(correct_cards) if correct_cards else "<p>No perfectly correct sentences found.</p>"

    return f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{_html.escape(title)}</title>
        {css_styles}
    </head>
    <body>
        <h2>{_html.escape(title)}</h2>
        {summary_panel}

        <h3 id="errors-section" class="section-heading error-title">Incorrect Predictions ({len(error_cards)} Sentences | Contains partially correct words)</h3>
        {errors_content}

        <h3 id="correct-section" class="section-heading correct-title">Perfect Predictions ({len(correct_cards)} Sentences | 100% words correct)</h3>
        {correct_content}
    </body>
    </html>"""


def export_single_analysis_report(processed_sentences, output_filename="model_analysis.html"):
    correct_cards = []
    error_cards = []

    total_error_words = 0
    total_correct_words = 0
    total_exact_correct_words = 0
    total_unreachable_words = 0

    total_ambiguous_words = 0
    correct_ambiguous_words = 0

    for idx, record in enumerate(processed_sentences):
        gold_idx_list = record["annotated-indices"]
        pred_idx_list = record["predicted-indices"]
        words_options = record["words-options"]

        word_count = len(words_options)
        has_error = False

        for i, options in enumerate(words_options):
            g = gold_idx_list[i]
            p = pred_idx_list[i]

            # A word is correct only if gold was findable at all.  -1 == -1
            # used to be scored as a hit here, inflating accuracy by exactly
            # the number of analyser misses.
            hit = (g >= 0 and g == p)

            if g < 0:
                total_unreachable_words += 1
            if hit:
                total_exact_correct_words += 1
            else:
                has_error = True

            cleaned_options = [opt for opt in options if not is_none_option(opt)]
            if len(cleaned_options) > 1:
                total_ambiguous_words += 1
                if hit:
                    correct_ambiguous_words += 1

        card_html = generate_sentence_html(record, idx)

        if has_error:
            error_cards.append(card_html)
            total_error_words += word_count
        else:
            correct_cards.append(card_html)
            total_correct_words += word_count

    full_html = build_combined_html(
        error_cards=error_cards,
        correct_cards=correct_cards,
        total_error_words=total_error_words,
        total_correct_words=total_correct_words,
        total_exact_correct_words=total_exact_correct_words,
        total_ambiguous_words=total_ambiguous_words,
        correct_ambiguous_words=correct_ambiguous_words,
        total_unreachable_words=total_unreachable_words,
        title="Model Performance & Disambiguation Analysis",
    )

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Successfully generated report with ambiguous word metrics: {output_filename}")

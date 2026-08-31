
import itertools
# import numpy as np
# import torch


# def validation_analysis(
#         val_groups,
#         model,
#         device,
#         morphology_generator,
#         vectorize_func,
#         window_size,
#         feature_dim,
#         eval_batch_size=16384 # chunk/batch size for safety
#     ):
#     model.eval()
#     sentence_details = []

#     print("wait for 5 minutes atleast...")

#     for group in val_groups:
#         sent_text = group['sentence']
#         sentence_options = morphology_generator(sent_text)

#         vectorized_options = []
#         for word_options in sentence_options:
#             word_choices = []
#             for opt in word_options:
#                 v = vectorize_func(opt)
#                 word_choices.append((v, opt)) # Store as (vector, raw_label)
#             vectorized_options.append(word_choices)

#         # 3. Create all possible sequence combinations (as an iterator, don't cast to list yet!)
#         all_possible_combinations = itertools.product(*vectorized_options)
        
#         padding_vec = np.zeros(feature_dim, dtype=np.float32)
        
#         best_score = -1.0
#         best_combo_labels = None
        
#         # We will manually batch the combinations to keep memory flat
#         current_chunk_vectors = []
#         current_chunk_labels = []
        
#         for combo in all_possible_combinations:
#             seq_vectors = [item[0] for item in combo]
#             seq_labels = [item[1] for item in combo]
            
#             seq_len = len(seq_vectors)
#             if seq_len > window_size:
#                 seq_vectors = seq_vectors[:window_size]
#                 seq_labels = seq_labels[:window_size]
#                 seq_len = window_size

#             padded = list(seq_vectors) + ([padding_vec] * (window_size - seq_len))
#             # current_chunk_vectors.append(np.stack(padded))
#             current_chunk_vectors.append(torch.from_numpy(np.array(padded, dtype=np.float32)))
#             current_chunk_labels.append(seq_labels)
            
#             # Once the chunk is full, evaluate it
#             if len(current_chunk_vectors) == eval_batch_size:
#                 best_score, best_combo_labels = _evaluate_chunk(
#                     current_chunk_vectors, current_chunk_labels, 
#                     model, device, best_score, best_combo_labels
#                 )
#                 current_chunk_vectors, current_chunk_labels = [], [] # Clear memory
                
#         # Process any remaining combinations left in the final partial chunk
#         if current_chunk_vectors:
#             best_score, best_combo_labels = _evaluate_chunk(
#                 current_chunk_vectors, current_chunk_labels, 
#                 model, device, best_score, best_combo_labels
#             )

#         # 6. Find Winner Indices
#         annotated_idx = []
#         predicted_idx = []
#         for i, word_options in enumerate(sentence_options):
#             for idx, option in enumerate(word_options):
#                 if group['annotated_list'][i] == option:
#                     annotated_idx.append(idx)

#                 if best_combo_labels and best_combo_labels[i] == option:
#                     predicted_idx.append(idx)

#         sentence_details.append({
#             "words-options": sentence_options,
#             "annotated-indices": annotated_idx,
#             "predicted-indices": predicted_idx
#         })

#         if len(annotated_idx) != len(sentence_options):
#             print("\nANNOTATED MISMATCH")
#             print("Sentence:", sent_text)

#             for i, word_options in enumerate(sentence_options):
#                 target = group['annotated_list'][i]

#                 found = target in word_options

#                 print(f"\nWord {i}")
#                 print("Target:", repr(target))
#                 print("Options:", [repr(x) for x in word_options])
#                 print("Found:", found)

#         if len(predicted_idx) != len(sentence_options):
#             print("\nPREDICTED MISMATCH")
#             print("Sentence:", sent_text)

#             for i, word_options in enumerate(sentence_options):
#                 pred = best_combo_labels[i]

#                 found = pred in word_options

#                 print(f"\nWord {i}")
#                 print("Predicted:", repr(pred))
#                 print("Options:", [repr(x) for x in word_options])
#                 print("Found:", found)

#     return sentence_details


# def _evaluate_chunk(chunk_vectors, chunk_labels, model, device, best_score, best_combo_labels):
#     """Helper function to process a single small batch safely."""
#     # Force float32 right here to instantly cut memory usage in half
#     # input_tensor = torch.tensor(np.array(chunk_vectors, dtype=np.float32), device=device)
#     input_tensor = torch.stack(chunk_vectors).to(device)
#     mask = torch.all(input_tensor == 0, dim=-1)
    
#     with torch.no_grad():
#         logits = model(input_tensor, mask=mask).squeeze(1)
#         scores = torch.sigmoid(logits).cpu().numpy()
        
#     chunk_best_idx = np.argmax(scores)
#     chunk_best_score = scores[chunk_best_idx]
    
#     # If this chunk beats the all-time high score for this sentence, track it
#     if chunk_best_score > best_score:
#         best_score = chunk_best_score
#         best_combo_labels = chunk_labels[chunk_best_idx]
        
#     return best_score, best_combo_labels


import heapq
import numpy as np
import pandas as pd
import torch


# ============================================================
# Configuration
# ============================================================

TOP_K = 250_000


# ============================================================
# Load bigram probabilities
# ============================================================

def load_bigram_probabilities(bigram_file):
    """
    Load:

        row    = previous tag
        column = next tag

    Returns:

        bigram_probs[previous_tag][next_tag] = probability
    """

    df = pd.read_csv(
        bigram_file,
        index_col=0
    )

    df = df.astype(float)

    bigram_probs = {}

    for previous_tag in df.index:

        bigram_probs[previous_tag] = {}

        for next_tag in df.columns:

            probability = df.loc[
                previous_tag,
                next_tag
            ]

            if probability > 0:
                bigram_probs[previous_tag][next_tag] = (
                    probability
                )

    return bigram_probs


# ============================================================
# Extract tags from one morphological option
# ============================================================

def get_option_tags(option):
    """
    Option format:

        [
            ['morpheme_1', 'tag_1'],
            ['morpheme_2', 'tag_2'],
            ...
        ]

    Returns:

        ['tag_1', 'tag_2', ...]
    """

    result = []

    for morpheme_tag in option:

        if (
            isinstance(morpheme_tag, (list, tuple))
            and len(morpheme_tag) >= 2
        ):
            result.append(morpheme_tag[1])

    return result


# ============================================================
# Transition log probability
# ============================================================

def transition_log_probability(
    previous_tag,
    current_tag,
    bigram_probs
):
    """
    Return log P(current_tag | previous_tag).

    If the transition is missing from the matrix,
    return 0.0, meaning that the transition is ignored.
    """

    if previous_tag is None:
        return 0.0

    if previous_tag not in bigram_probs:
        return 0.0

    if current_tag not in bigram_probs[previous_tag]:
        return 0.0

    probability = bigram_probs[
        previous_tag
    ][
        current_tag
    ]

    if probability <= 0:
        return 0.0

    return float(np.log(probability))


# ============================================================
# Precompute information for every morphological option
# ============================================================

def prepare_options(
    sentence_options,
    bigram_probs
):
    """
    For every option, precompute:

        - vectors
        - raw option
        - tag sequence
        - first tag
        - last tag
        - internal log probability

    The internal probability is:

        log P(t2|t1)
        + log P(t3|t2)
        + ...
    """

    prepared = []

    for word_options in sentence_options:

        word_prepared = []

        for option in word_options:

            tags = get_option_tags(option)

            # -----------------------------------------------
            # Internal probability of this word's morphology
            # -----------------------------------------------

            internal_score = 0.0

            for i in range(1, len(tags)):

                internal_score += (
                    transition_log_probability(
                        tags[i - 1],
                        tags[i],
                        bigram_probs
                    )
                )

            first_tag = (
                tags[0]
                if tags
                else None
            )

            last_tag = (
                tags[-1]
                if tags
                else None
            )

            word_prepared.append({
                "option": option,
                "tags": tags,
                "first_tag": first_tag,
                "last_tag": last_tag,
                "internal_score": internal_score
            })

        prepared.append(word_prepared)

    return prepared


# ============================================================
# TOP-K dynamic programming
# ============================================================

def get_top_probability_candidates(
    sentence_options,
    bigram_probs,
    k=TOP_K
):
    """
    Find the top-K complete morphological-analysis sequences
    according to the empirical bigram probabilities.

    This DOES NOT enumerate the full Cartesian product.

    Each state contains:

        score
        sequence of option indices
        last tag

    Only the best K partial sequences are retained.

    Returns:

        [
            (score, option_index_sequence),
            ...
        ]
    """

    prepared = prepare_options(
        sentence_options,
        bigram_probs
    )

    # --------------------------------------------------------
    # State representation:
    #
    # (score, option_indices, last_tag)
    # --------------------------------------------------------

    states = [
        (
            0.0,
            tuple(),
            None
        )
    ]

    # ========================================================
    # Process one word at a time
    # ========================================================

    for word_index, word_options in enumerate(
        prepared
    ):

        print(
            f"Processing word "
            f"{word_index + 1}/"
            f"{len(prepared)} "
            f"({len(word_options)} options)"
        )

        new_states = []

        # ----------------------------------------------------
        # Expand existing states with every option of this
        # word.
        # ----------------------------------------------------

        for (
            previous_score,
            previous_indices,
            previous_last_tag
        ) in states:

            for option_index, option in enumerate(
                word_options
            ):

                # --------------------------------------------
                # Transition from previous word into this word
                # --------------------------------------------

                transition_score = (
                    transition_log_probability(
                        previous_last_tag,
                        option["first_tag"],
                        bigram_probs
                    )
                )

                score = (
                    previous_score
                    + transition_score
                    + option["internal_score"]
                )

                new_indices = (
                    previous_indices
                    + (option_index,)
                )

                new_states.append(
                    (
                        score,
                        new_indices,
                        option["last_tag"]
                    )
                )

        # ----------------------------------------------------
        # Keep only top-K states.
        #
        # IMPORTANT:
        # We group by last tag before pruning.
        # ----------------------------------------------------

        states_by_last_tag = {}

        for state in new_states:

            score = state[0]
            last_tag = state[2]

            if last_tag not in states_by_last_tag:
                states_by_last_tag[last_tag] = []

            states_by_last_tag[last_tag].append(
                state
            )

        states = []

        # ----------------------------------------------------
        # Keep K best states globally, while ensuring that
        # every possible last-tag state survives.
        # ----------------------------------------------------

        for tag_states in states_by_last_tag.values():

            if len(tag_states) > k:

                tag_states = heapq.nlargest(
                    k,
                    tag_states,
                    key=lambda x: x[0]
                )

            states.extend(tag_states)

        # ----------------------------------------------------
        # Global top-K pruning
        # ----------------------------------------------------

        if len(states) > k:

            states = heapq.nlargest(
                k,
                states,
                key=lambda x: x[0]
            )

        print(
            f"  retained states: "
            f"{len(states):,}"
        )

    # ========================================================
    # Convert final states
    # ========================================================

    states.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        (
            score,
            option_indices
        )
        for (
            score,
            option_indices,
            _
        ) in states[:k]
    ]


# ============================================================
# Convert top-K index sequences into model candidates
# ============================================================

def materialize_top_candidates(
    top_candidates,
    vectorized_options
):
    """
    Convert:

        (score, option_index_sequence)

    into:

        (probability_score, combo)

    where combo has the same structure as the original
    itertools.product() output.
    """

    result = []

    for probability_score, indices in top_candidates:

        combo = tuple(
            vectorized_options[word_index][
                option_index
            ]
            for word_index, option_index
            in enumerate(indices)
        )

        result.append(
            (
                probability_score,
                combo
            )
        )

    return result


# ============================================================
# Evaluate Transformer batch
# ============================================================

def _evaluate_chunk(
    chunk_vectors,
    chunk_labels,
    model,
    device,
    best_score,
    best_combo_labels
):

    input_tensor = torch.stack(
        chunk_vectors
    ).to(
        device,
        non_blocking=True
    )

    mask = torch.all(
        input_tensor == 0,
        dim=-1
    )

    with torch.no_grad():

        logits = model(
            input_tensor,
            mask=mask
        ).squeeze(1)

        # No sigmoid required for ranking.
        chunk_best_idx = torch.argmax(
            logits
        ).item()

        chunk_best_score = logits[
            chunk_best_idx
        ].item()

    if chunk_best_score > best_score:

        best_score = chunk_best_score

        best_combo_labels = (
            chunk_labels[
                chunk_best_idx
            ]
        )

    return (
        best_score,
        best_combo_labels
    )


# ============================================================
# Evaluate top-K candidates
# ============================================================

def evaluate_top_candidates(
    top_candidates,
    vectorized_options,
    model,
    device,
    window_size,
    feature_dim,
    eval_batch_size
):

    padding_vec = np.zeros(
        feature_dim,
        dtype=np.float32
    )

    current_vectors = []
    current_labels = []

    best_score = -float("inf")
    best_combo_labels = None

    for _, combo in top_candidates:

        seq_vectors = [
            item[0]
            for item in combo
        ]

        seq_labels = [
            item[1]
            for item in combo
        ]

        # ----------------------------------------------------
        # Truncate
        # ----------------------------------------------------

        if len(seq_vectors) > window_size:

            seq_vectors = (
                seq_vectors[:window_size]
            )

            seq_labels = (
                seq_labels[:window_size]
            )

        # ----------------------------------------------------
        # Pad
        # ----------------------------------------------------

        padded = (
            list(seq_vectors)
            + [padding_vec] *
            (
                window_size
                - len(seq_vectors)
            )
        )

        current_vectors.append(
            torch.from_numpy(
                np.asarray(
                    padded,
                    dtype=np.float32
                )
            )
        )

        current_labels.append(
            seq_labels
        )

        # ----------------------------------------------------
        # Full batch
        # ----------------------------------------------------

        if len(current_vectors) >= eval_batch_size:

            (
                best_score,
                best_combo_labels
            ) = _evaluate_chunk(
                current_vectors,
                current_labels,
                model,
                device,
                best_score,
                best_combo_labels
            )

            current_vectors = []
            current_labels = []

    # --------------------------------------------------------
    # Final partial batch
    # --------------------------------------------------------

    if current_vectors:

        (
            best_score,
            best_combo_labels
        ) = _evaluate_chunk(
            current_vectors,
            current_labels,
            model,
            device,
            best_score,
            best_combo_labels
        )

    return (
        best_score,
        best_combo_labels
    )


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
    bigram_file="annotation/empirical-bigram-probabilities.csv",
    max_probability_candidates=TOP_K,
    candidate_stats_file="sentence_error_analysis.csv"
):

    model.eval()

    sentence_details = []
    candidate_statistics = []

    # --------------------------------------------------------
    # Load probabilities once
    # --------------------------------------------------------

    print(
        "Loading empirical bigram probabilities..."
    )

    bigram_probs = load_bigram_probabilities(
        bigram_file
    )

    print(
        f"Loaded {len(bigram_probs):,} "
        f"previous-tag entries."
    )

    # ========================================================
    # Sentences
    # ========================================================

    for group_number, group in enumerate(
        val_groups,
        start=1
    ):

        sent_text = group["sentence"]

        print("\n" + "=" * 70)

        print(
            f"Sentence {group_number}/"
            f"{len(val_groups)}"
        )

        print(
            sent_text
        )

        # ====================================================
        # Generate options
        # ====================================================

        sentence_options = (
            morphology_generator(
                sent_text
            )
        )


        # ====================================================
        # Sentence level statistics
        # ====================================================
        sentence_length = len(sentence_options)
        # Number of valid morphological analyses per word
        option_counts = []

        for word_options in sentence_options:

            valid_options = [
                opt
                for opt in word_options
                if opt != "None of the above"
            ]

            option_counts.append(len(valid_options))

        # Number of ambiguous words
        ambiguous_words = sum(
            count > 1
            for count in option_counts
        )

        # Total candidate analyses for the sentence
        #
        # Python's normal int has arbitrary precision, so this
        # can safely handle values such as 10^25, 10^50, etc.
        total_candidate_analyses = 1

        for count in option_counts:
            total_candidate_analyses *= count

        # ====================================================
        # Vectorize
        # ====================================================

        vectorized_options = []

        for word_options in sentence_options:

            word_choices = []

            for opt in word_options:

                v = vectorize_func(opt)

                word_choices.append(
                    (
                        v,
                        opt
                    )
                )

            vectorized_options.append(
                word_choices
            )

        # ====================================================
        # Calculate candidate count
        # ====================================================

        total_candidates = 1

        for options in vectorized_options:

            total_candidates *= len(
                options
            )

        print(
            f"Total possible sequences: "
            f"{total_candidates:,}"
        )

        # ====================================================
        # CASE 1:
        # Exhaustive inference
        # ====================================================

        if total_candidates <= max_probability_candidates:

            print(
                "Inference mode: "
                "EXHAUSTIVE"
            )

            all_combinations = itertools.product(
                *vectorized_options
            )

            padding_vec = np.zeros(
                feature_dim,
                dtype=np.float32
            )

            current_vectors = []
            current_labels = []

            best_score = -float("inf")
            best_combo_labels = None

            for combo in all_combinations:

                seq_vectors = [
                    item[0]
                    for item in combo
                ]

                seq_labels = [
                    item[1]
                    for item in combo
                ]

                if len(seq_vectors) > window_size:

                    seq_vectors = (
                        seq_vectors[:window_size]
                    )

                    seq_labels = (
                        seq_labels[:window_size]
                    )

                padded = (
                    list(seq_vectors)
                    + [padding_vec] *
                    (
                        window_size
                        - len(seq_vectors)
                    )
                )

                current_vectors.append(
                    torch.from_numpy(
                        np.asarray(
                            padded,
                            dtype=np.float32
                        )
                    )
                )

                current_labels.append(
                    seq_labels
                )

                if (
                    len(current_vectors)
                    >= eval_batch_size
                ):

                    (
                        best_score,
                        best_combo_labels
                    ) = _evaluate_chunk(
                        current_vectors,
                        current_labels,
                        model,
                        device,
                        best_score,
                        best_combo_labels
                    )

                    current_vectors = []
                    current_labels = []

            if current_vectors:

                (
                    best_score,
                    best_combo_labels
                ) = _evaluate_chunk(
                    current_vectors,
                    current_labels,
                    model,
                    device,
                    best_score,
                    best_combo_labels
                )

        # ====================================================
        # CASE 2:
        # TOP-K PROBABILITY PRUNING
        # ====================================================

        else:

            print(
                "Inference mode: "
                "TOP-K BIGRAM PROBABILITY"
            )

            print(
                f"Generating top "
                f"{max_probability_candidates:,} "
                f"probability candidates..."
            )

            top_candidates = (
                get_top_probability_candidates(
                    sentence_options,
                    bigram_probs,
                    k=max_probability_candidates
                )
            )

            print(
                f"Generated "
                f"{len(top_candidates):,} "
                f"candidates."
            )

            # ------------------------------------------------
            # Materialize their vectors/labels
            # ------------------------------------------------

            top_candidates = (
                materialize_top_candidates(
                    top_candidates,
                    vectorized_options
                )
            )

            print(
                "Running Transformer inference..."
            )

            (
                best_score,
                best_combo_labels
            ) = evaluate_top_candidates(
                top_candidates,
                vectorized_options,
                model,
                device,
                window_size,
                feature_dim,
                eval_batch_size
            )

        # ====================================================
        # Gold / predicted indices
        # ====================================================

        annotated_idx = []
        predicted_idx = []

        for i, word_options in enumerate(sentence_options):

            # ========================================================
            # GOLD INDEX
            # ========================================================

            gold_option = group["annotated_list"][i]

            try:
                gold_index = word_options.index(gold_option)
            except ValueError:
                gold_index = -1

                print("\nANNOTATED MISMATCH")
                print("Sentence:", sent_text)
                print("Word:", i)
                print("Gold:", repr(gold_option))
                print(
                    "Options:",
                    [repr(x) for x in word_options]
                )

            annotated_idx.append(gold_index)

            # ========================================================
            # PREDICTED INDEX
            # ========================================================

            if best_combo_labels is None:
                predicted_idx.append(-1)

            else:

                # This protects against an unexpectedly short
                # predicted sequence.
                if i >= len(best_combo_labels):

                    predicted_idx.append(-1)

                    print("\nPREDICTED LENGTH MISMATCH")
                    print("Sentence:", sent_text)
                    print(
                        f"Word {i}: prediction is missing"
                    )

                else:

                    predicted_option = best_combo_labels[i]

                    try:
                        predicted_index = (
                            word_options.index(
                                predicted_option
                            )
                        )

                    except ValueError:

                        predicted_index = -1

                        print("\nPREDICTED MISMATCH")
                        print("Sentence:", sent_text)
                        print("Word:", i)
                        print(
                            "Predicted:",
                            repr(predicted_option)
                        )
                        print(
                            "Options:",
                            [
                                repr(x)
                                for x in word_options
                            ]
                        )

                    predicted_idx.append(
                        predicted_index
                    )

        # ====================================================
        # Save sentence result
        # ====================================================

        sentence_details.append({
            "words-options": sentence_options,
            "annotated-indices": annotated_idx,
            "predicted-indices": predicted_idx
        })

        # ====================================================
        # Diagnostics
        # ====================================================

        if (
            len(annotated_idx)
            != len(sentence_options)
        ):

            print(
                "\nANNOTATED MISMATCH"
            )

            for i, word_options in enumerate(
                sentence_options
            ):

                target = (
                    group["annotated_list"][i]
                )

                print(
                    f"Word {i}: "
                    f"target={repr(target)}"
                )

                print(
                    "Options:",
                    [
                        repr(x)
                        for x in word_options
                    ]
                )

        if (
            best_combo_labels is not None
            and len(predicted_idx)
            != len(sentence_options)
        ):

            print(
                "\nPREDICTED MISMATCH"
            )

            for i, word_options in enumerate(
                sentence_options
            ):

                pred = best_combo_labels[i]

                print(
                    f"Word {i}: "
                    f"prediction={repr(pred)}"
                )

                print(
                    "Options:",
                    [
                        repr(x)
                        for x in word_options
                    ]
                )
        # =====================================================
        # Number of incorrect words
        # =====================================================

        incorrect_words = sum(
            g != p
            for g, p in zip(
                annotated_idx,
                predicted_idx
            )
        )
        # =====================================================
        # Save sentence-level statistics
        # =====================================================

        candidate_statistics.append({
            "sentence": sent_text,
            "length": sentence_length,
            "ambiguous_words": ambiguous_words,
            "incorrect_words": incorrect_words,
            "candidate_analyses": total_candidate_analyses
        })


    # =========================================================
    # Save sentence-level statistics
    # =========================================================

    stats_df = pd.DataFrame(
        candidate_statistics
    )

    stats_df.to_csv(
        candidate_stats_file,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nSaved sentence-level error analysis to: "
        f"{candidate_stats_file}"
    )
    return sentence_details


def generate_sentence_html(record, index):
    """Generates the HTML block for a single sentence using only your specified keys."""
    sent_id = record.get("sentence-id", f"Batch Index: {index}")
    sentence_text = record.get("sentence", "")
    
    words_options = record["words-options"]
    annotated_indices = record["annotated-indices"]
    predicted_indices = record["predicted-indices"]
    
    html = f"""
    <div class="card">
        <div class="card-header">Sentence ID: {sent_id}</div>
        <div class="card-body">
            {f'<p class="sentence-text">"{sentence_text}"</p>' if sentence_text else ''}
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
                style_class = "badge badge-coincide"   # Green
            elif idx == gold:
                style_class = "badge badge-annotated"  # Blue
            elif idx == pred:
                style_class = "badge badge-predicted"  # Yellow
            else:
                style_class = "badge-none"
                
            opt_html_list.append(f'<span class="{style_class}">{opt}</span>')
            
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
    title
):
    """Styles and bundles both categories into one clean HTML file with ambiguous word accuracy tracking."""
    css_styles = """
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }
        h2 { color: #2c3e50; margin-bottom: 5px; }
        .section-heading { color: #1e293b; padding-bottom: 10px; border-bottom: 2px solid #cbd5e1; margin-top: 40px; margin-bottom: 20px; }
        .error-title { color: #dc3545; border-color: #f5c6cb; }
        .correct-title { color: #28a745; border-color: #c3e6cb; }
        
        /* Dashboard & Legend Styles */
        .dashboard { display: flex; gap: 20px; margin-bottom: 25px; flex-wrap: wrap; }
        .panel { background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; flex: 1; min-width: 250px; }
        .metric-value { font-size: 1.15em; font-weight: bold; color: #0f172a; margin-top: 5px; line-height: 1.5; }
        .legend-item { display: inline-block; margin-right: 25px; font-weight: bold; font-size: 0.9em; }
        .nav-links a { display: inline-block; margin-right: 15px; color: #007bff; font-weight: bold; text-decoration: none; }
        .nav-links a:hover { text-decoration: underline; }
        
        /* Sentence Card Styles */
        .card { background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; border: 1px solid #e2e8f0; overflow: hidden; }
        .card-header { background-color: #f1f5f9; padding: 12px 20px; font-weight: bold; color: #475569; font-size: 0.9em; border-bottom: 1px solid #e2e8f0; }
        .card-body { padding: 20px; }
        .sentence-text { font-style: italic; font-size: 1.1em; color: #1e293b; margin-top: 0; margin-bottom: 15px; }
        
        /* Table Styles */
        table { width: 100%; border-collapse: collapse; }
        th { background-color: #f8fafc; text-align: left; padding: 10px; font-size: 0.85em; color: #64748b; border-bottom: 2px solid #cbd5e1; text-transform: uppercase; }
        td { padding: 10px; border-bottom: 1px solid #f1f5f9; font-size: 0.95em; }
        .pos-cell { font-weight: bold; color: #64748b; }
        .divider { color: #cbd5e1; margin: 0 6px; }
        
        /* Color Badges */
        .badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; }
        .badge-coincide { background-color: #28a745; color: white; } /* Green */
        .badge-annotated { background-color: #007bff; color: white; } /* Blue */
        .badge-predicted { background-color: #ffc107; color: black; } /* Yellow */
        .badge-none { color: #333333; padding: 4px 8px; }
    </style>
    """
    
    total_sentences = len(error_cards) + len(correct_cards)
    total_words = total_error_words + total_correct_words
    
    # Accuracy Calculations
    sentence_accuracy = (len(correct_cards) / total_sentences * 100) if total_sentences > 0 else 0.0
    word_accuracy = (total_exact_correct_words / total_words * 100) if total_words > 0 else 0.0
    ambiguous_accuracy = (correct_ambiguous_words / total_ambiguous_words * 100) if total_ambiguous_words > 0 else 0.0

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
        <title>{title}</title>
        {css_styles}
    </head>
    <body>
        <h2>{title}</h2>
        {summary_panel}
        
        <h3 id="errors-section" class="section-heading error-title">Incorrect Predictions ({len(error_cards)} Sentences | Contains partially correct words)</h3>
        {errors_content}
        
        <h3 id="correct-section" class="section-heading correct-title">Perfect Predictions ({len(correct_cards)} Sentences | 100% words correct)</h3>
        {correct_content}
    </body>
    </html>"""

def export_single_analysis_report(processed_sentences, output_filename="model_analysis.html"):
    """Evaluates sentences, tracks overall and ambiguous token hits, and renders the layout."""
    correct_cards = []
    error_cards = []
    
    total_error_words = 0
    total_correct_words = 0
    total_exact_correct_words = 0
    
    # New counters for tracking ambiguous word performance
    total_ambiguous_words = 0
    correct_ambiguous_words = 0
    
    for idx, record in enumerate(processed_sentences):
        gold_idx_list = record["annotated-indices"]
        pred_idx_list = record["predicted-indices"]
        words_options = record["words-options"]
        
        word_count = len(words_options)
        
        # Track word-level evaluations
        for i, options in enumerate(words_options):
            g = gold_idx_list[i]
            p = pred_idx_list[i]
            
            # 1. Global word hit tracking
            if g == p:
                total_exact_correct_words += 1
            
            # 2. Check if this specific word is ambiguous (ignoring 'None of the above')
            cleaned_options = [opt for opt in options if opt != 'None of the above']
            if len(cleaned_options) > 1:
                total_ambiguous_words += 1
                if g == p:
                    correct_ambiguous_words += 1
        
        has_error = any(g != p for g, p in zip(gold_idx_list, pred_idx_list))
        card_html = generate_sentence_html(record, idx)
        
        if has_error:
            error_cards.append(card_html)
            total_error_words += word_count
        else:
            correct_cards.append(card_html)
            total_correct_words += word_count
            
    # Compile document with all tracking parameters mapped out
    full_html = build_combined_html(
        error_cards=error_cards, 
        correct_cards=correct_cards, 
        total_error_words=total_error_words, 
        total_correct_words=total_correct_words, 
        total_exact_correct_words=total_exact_correct_words,
        total_ambiguous_words=total_ambiguous_words,
        correct_ambiguous_words=correct_ambiguous_words,
        title="Model Performance & Disambiguation Analysis"
    )
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Successfully generated report with ambiguous word metrics: {output_filename}")
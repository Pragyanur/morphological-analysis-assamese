import json
from collections import defaultdict
import numpy as np
import re


with open("resources/PoS-mapping.json", 'r', encoding='utf-8') as f:
    POS_MAP = json.load(f)

with open("resources/suffix-all-features.json", "r", encoding='utf-8') as f:
    morpheme_all_features = json.load(f)

with open("resources/derivational_probabilities_type1.json", "r", encoding='utf-8') as f:
    DERIVATIONS_T1 = json.load(f)

with open("resources/pronouns-case.json", "r", encoding='utf-8') as f:
    PRONOUN_DETAIL = json.load(f)

with open("resources/tag-seq-rules-updated.json", "r", encoding='utf-8') as f:
    TAG_SEQ_RULES = json.load(f)

with open("resources/suffix-number.json", "r", encoding='utf-8') as f:
    NUMBER_DETAIL = json.load(f)

with open("resources/vowel_verbs.json", "r", encoding='utf-8') as f:
    VOWEL_VERB = json.load(f)

with open("resources/verbable.json", "r", encoding='utf-8') as f:
    VERBABLE = json.load(f)

with open("resources/symbol-mapping.json", "r", encoding='utf-8') as f:
    SYMBOL_MAP = json.load(f)

features = set()

for m,d in morpheme_all_features.items():
    features.update(d)


ID_TO_LABEL = {}
LABEL_TO_ID = {}

ID_TO_WORD = {}
WORD_TO_ID = {}

# sort the data
features = sorted(features)

for idx, feature in enumerate(features):
    LABEL_TO_ID[feature] = idx
    ID_TO_LABEL[idx] = feature

# absolutive case, which is only integrated with pronoun
LABEL_TO_ID["abs"] = len(LABEL_TO_ID)
ID_TO_LABEL[len(LABEL_TO_ID)] = "abs"
# word start tag
LABEL_TO_ID["ws"] = len(LABEL_TO_ID)
ID_TO_LABEL[len(LABEL_TO_ID)] = "ws"
# word end tag
LABEL_TO_ID["we"] = len(LABEL_TO_ID)
ID_TO_LABEL[len(LABEL_TO_ID)] = "we"

for idx, word in enumerate(sorted(morpheme_all_features.keys())):
    WORD_TO_ID[word] = idx
    ID_TO_WORD[idx] = word

WORD_TO_ID[" "] = len(WORD_TO_ID)
ID_TO_WORD[len(WORD_TO_ID)] = " "

# initialization
MAX_LABELS = len(LABEL_TO_ID)
LOWEST_DERIVATIONAL_PROBABILITY = 0.001

def derive_PoS(PoS_tag, morpheme, derivations_type1=DERIVATIONS_T1, alpha=LOWEST_DERIVATIONAL_PROBABILITY):
    all_der_list = derivations_type1[morpheme]
    max_prob = 0
    most_porb = PoS_tag
    for i in all_der_list:
        if i[0][0] == PoS_tag:
            if i[1] > max_prob:
                max_prob = i[1]
                most_porb = i[0][1]
                return most_porb, max_prob
    return most_porb, alpha

def integratedCasePronoun(word, pronoun_features=PRONOUN_DETAIL):
    if word not in pronoun_features:
        return "Pronoun"
    else:
        for i in pronoun_features[word]:
            if i in ["erg", "acc", "dat", "loc", "abs", "gen"]:
                return i
    return "Pronoun"

# Create the reverse mapping for indices to labels
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

morphemes_by_tag = defaultdict(set)
# Assuming your old all_morphemes was { "morpheme": "Tag" }
for morph, tag in morpheme_all_features.items():
    for t in tag:
        morphemes_by_tag[t].add(morph)

def rule_matrix_init(label_to_id=LABEL_TO_ID, bigram_tag_rules=TAG_SEQ_RULES):
    rule_matrix = np.zeros((len(label_to_id),len(label_to_id)))
    for i in label_to_id:
        for j in label_to_id:
            if i == "ws":
                # not in
                if j not in bigram_tag_rules["ws"]:
                    rule_matrix[label_to_id[i]][label_to_id[j]] = 1
            elif j in bigram_tag_rules[i]:
                rule_matrix[label_to_id[i]][label_to_id[j]] = 1

    return rule_matrix

rule_matrix = rule_matrix_init()

def rule_based_forward_analysis(word, morphemes_by_tag=morphemes_by_tag, rule_matrix=rule_matrix, label_to_id=LABEL_TO_ID, id_to_label=ID_TO_LABEL):
    all_valid_sequences = []
    
    # # Use memoization to prevent redundant calculations and kernel crashes
    # memo = {}

    def solve(remaining_word, current_tag, current_sequence):
        # 1. BASE CASE: Word is fully exhausted
        if not remaining_word:
            # Check if the current_tag is allowed to end a word (transition to 'we')
            if rule_matrix[label_to_id[current_tag]][label_to_id.get("we", -1)] > 0:
                # Check for multiple derivations
                if sum(i[1] == "derivational" for i in current_sequence) <= 1:
                    all_valid_sequences.append([item[:] for item in current_sequence])
            return

        # 2. Get ALL possible next tags from the rule matrix
        valid_next_tags = []
        current_tag_id = label_to_id[current_tag]
        for tag_id, value in enumerate(rule_matrix[current_tag_id]):
            if value > 0:
                tag_name = id_to_label[tag_id]
                if tag_name != "we": # 'we' is handled in the base case
                    valid_next_tags.append(tag_name)
        
        # 3. Try every possible morpheme length
        for length in range(1, len(remaining_word) + 1):
            morpheme = remaining_word[:length]
            suffix = remaining_word[length:]

            for tag_name in valid_next_tags:
                universe = morphemes_by_tag.get(tag_name, set())

                if morpheme in universe:
                    branch_sequence = [item[:] for item in current_sequence]
                    next_state_tag = tag_name

                    # --- Logic: Derivational ---
                    if tag_name == "derivational" and len(branch_sequence) > 0:
                        # Ensure derive_PoS exists in your namespace
                        most_probable_derivation, _ = derive_PoS(branch_sequence[-1][1], morpheme)
                        branch_sequence.append([morpheme, tag_name]) # Keep 'derivational' and the result
                        branch_sequence.append(["", most_probable_derivation])
                        next_state_tag = most_probable_derivation
                    
                    # --- Logic: Pronoun ---
                    elif tag_name == "Pronoun":
                        branch_sequence.append([morpheme, "Pronoun"])
                        case = integratedCasePronoun(morpheme)
                        if case != "Pronoun":
                            branch_sequence.append(["", case])
                            next_state_tag = case
                    
                    # --- Standard Case ---
                    else:
                        branch_sequence.append([morpheme, tag_name])

                    # 4. Recursion
                    solve(suffix, next_state_tag, branch_sequence)

    # Start the analysis
    solve(word, "ws", [])

    return sorted(all_valid_sequences, key=lambda x: len(x))

SPECIAL_CHARS = "\",:;()[]{}/|`~@#$\\%^&*_+-=?!।"

def sentence_word_options(sentence, special_chars=SPECIAL_CHARS):
    sentence_options = []
    for w in sentence.split():
        opts = []
        opts = rule_based_forward_analysis(w)
        
        if opts:
            sentence_options.append(opts)
        elif w in special_chars:
            sentence_options.append([[[w, "symbol"]]])
        elif re.match(r"\d+", w):
            sentence_options.append([[[w, "numeric-string"]]])
        else:
            sentence_options.append([[[w, "Others"]]])
    return sentence_options

def pronoun_person_honorific(pronoun, pronoun_detail=PRONOUN_DETAIL):
    person = ["p1", "p2", "p3"] # for 1st person, 2nd person and 3rd person
    honorific = ["a", "i", "n", "f"] # for all, informal, neutral, formal
    p = ""
    h = ""
    if pronoun in pronoun_detail:
        for tag in pronoun_detail[pronoun]:
            if tag in person:
                p = tag
            if tag in honorific:
                h = tag
    return p, h

def tense_person_honorific(verb_suffix):
    t = ""
    p = ""
    h = ""
    tense_pat = r"future|past|present"
    person_pat = r"[123]"
    honorific_pat = r"[ainf]"
    match = re.search(tense_pat, verb_suffix)
    if match:
        t = match.group(0)
    match = re.search(person_pat, verb_suffix)
    if match:
        person = match.group(0)
        p = "p" + person
    match = re.search(honorific_pat, verb_suffix)
    if match:
        h = match.group(0)
    return t, p, h

VERB_SUFFIXES = [
    'future-1',
    'future-2f',
    'future-2n',
    'future-3a',
    'past-1',
    'past-2a',
    'past-2f',
    'past-3a',
    'present-1',
    'present-2f',
    'present-2i',
    'present-2n',
    'present-3a',
]

# 64 features
MORPHOLOGY_FEATURES = {
    'Abstract Noun': 0,
    'Adjective': 0,
    'Adjective Adj.': 0,
    'Adposition': 0,
    'Adverb': 0,
    'Common Noun': 0,
    'Conjuction': 0,
    'Conjunction': 0,
    'Interjection': 0,
    'Material Noun': 0,
    'Noun': 0,
    'Others': 0,
    'Pronoun': 0,
    'Proper Adj.': 0,
    'Proper Noun': 0,
    'Verb': 0,
    'Verb-Intran.': 0,
    'Verb-Trans.': 0,
    'Verbable': 0,
    'Verbal Adj.': 0,
    'Verbal Noun': 0,
    'a': 0,
    'abs': 0,
    'acc': 0,
    'adj_comp': 0,
    'after_comp': 0,
    'ardha': 0,
    'compare': 0,
    'dash': 0,
    'dat': 0,
    'derivational': 0,
    'emph': 0,
    'end_sym': 0,
    'f': 0,
    'future': 0,
    'gen': 0,
    'i': 0,
    'infix': 0,
    'inst': 0,
    'loc': 0,
    'manner': 0,
    'mod_attr': 0,
    'modal': 0,
    'n': 0,
    'neg_v_verb': 0,
    'negative': 0,
    'erg': 0,
    'number': 0,
    'numeric-string': 0,
    'numerical': 0,
    'other_sym': 0,
    'p1': 0,
    'p2': 0,
    'p3': 0,
    'past': 0,
    'pause_sym': 0,
    'pl': 0,
    'present': 0,
    'quantity': 0,
    'reason_comp': 0,
    'sg': 0,
    't_loc': 0,
    'when_comp': 0,
    'without': 0
}

def vectorize_morphology(
        sequence,
        pos_map=POS_MAP,
        num_detail=NUMBER_DETAIL,
        default_features=MORPHOLOGY_FEATURES,
        verb_suffixes=VERB_SUFFIXES,
        symbol_map=SYMBOL_MAP
        # vowel_verb_map=VOWEL_VERB
        # verbable_map=VERBABLE
    ):
    vector_dict = default_features.copy()
    pos_decay = 0.25
    pos_num = 0
    for morpheme in sequence[::-1]:
        this_tag = morpheme[1]
        this_morpheme = morpheme[0]
        if this_tag == "number":
            vector_dict["number"] = 1
            vector_dict[num_detail[this_morpheme]] += 1
        elif this_tag == "Pronoun":
            vector_dict["Pronoun"] += 1
            pos_num += 1
            p,h = pronoun_person_honorific(this_morpheme)
            if p != "": vector_dict[p] += 1
            if h != "": vector_dict[h] += 1
        elif this_tag == "symbol":
            vector_dict[symbol_map[this_morpheme]] += 1
        elif this_tag == "derivational":
            vector_dict[this_tag] = 1
        elif this_tag in pos_map:
            weight = 1 - pos_decay * pos_num
            if vector_dict[this_tag] == 0:
                vector_dict[this_tag] = weight
                pos_num += 1
            if vector_dict[pos_map[this_tag]] == 0:
                vector_dict[pos_map[this_tag]] = weight
                pos_num += 1
        elif this_tag in verb_suffixes:
            t,p,h = tense_person_honorific(this_tag)
            if t != "": vector_dict[t] += 1
            if p != "": vector_dict[p] += 1
            if h != "": vector_dict[h] += 1
        else: vector_dict[this_tag] += 1

    vector = np.array([value for key, value in sorted(vector_dict.items())])

    return vector

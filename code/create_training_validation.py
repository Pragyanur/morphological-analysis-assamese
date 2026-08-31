from itertools import combinations, product
import pandas as pd
import numpy as np
import random
import ast

from morphology import vectorize_morphology

def pad_with_jitter(sequence, window_size, padding_vec, feature_dim):
    """
    Places the sequence at a random offset. 
    Ensures every element in the list is a numpy array of shape (feature_dim,).
    """
    seq_len = len(sequence)
    
    # If the sentence is longer than the window, truncate it
    if seq_len > window_size:
        sequence = sequence[:window_size]
        seq_len = window_size
    
    slack = window_size - seq_len
    start_offset = random.randint(0, slack)
    # start_offset = int(np.random.beta(2, 5) * slack)
    end_offset = slack - start_offset
    
    # Create the padded list
    # Ensure the padding_vec is exactly the right shape
    padded_seq = ([padding_vec] * start_offset) + sequence + ([padding_vec] * end_offset)
    
    # Convert to a single numpy array of shape (40, 64)
    return np.stack(padded_seq).astype(np.float32)


def generate_k_swap_negatives(
    gold_sequence,
    alt_candidates_vectors,
    k
):
    changeable = [i for i, alts in enumerate(alt_candidates_vectors) if len(alts) > 0]

    if len(changeable) < k:
        return

    for positions in combinations(changeable, k):
        alt_lists = [alt_candidates_vectors[pos] for pos in positions]

        for chosen_alts in product(*alt_lists):
            neg_seq = list(gold_sequence)
            for pos, alt in zip(positions, chosen_alts):
                neg_seq[pos] = alt
            yield neg_seq

def create_complete_morphology_dataset(
    df,
    vectorize_func,
    window_size=64,
    feature_dim=64,
    neg_ratio=3,
    augment_factor=3
):
    X = []
    y = []
    sentence_ids = []
    padding_vec = np.zeros(feature_dim, dtype=np.float32)

    df = df.copy()
    df['selected-index'] = df['selected-index'].fillna(-1).astype(int)
    grouped = df.groupby('sentence-id')

    for sent_id, group in grouped:
        group = group.sort_values('word-id')
        gold_sequence = []
        alt_candidates_vectors = []
        is_sentence_valid = True

        for _, row in group.iterrows():
            try:
                options_list = ast.literal_eval(row['options'])
            except:
                is_sentence_valid = False
                break

            idx = row['selected-index']

            if (
                idx == -1
                or idx >= len(options_list)
                or options_list[idx] == 'None of the above'
            ):
                is_sentence_valid = False
                break

            gold_vec = (
                np.array(vectorize_func(options_list[idx]))
                .flatten()
                .astype(np.float32)
            )

            if gold_vec.shape[0] != feature_dim:
                is_sentence_valid = False
                break

            gold_sequence.append(gold_vec)
            wrongs = []

            for i, opt in enumerate(options_list):
                if opt == 'None of the above':
                    continue
                vec = (
                    np.array(vectorize_func(opt))
                    .flatten()
                    .astype(np.float32)
                )
                if vec.shape[0] != feature_dim:
                    continue
                if not np.array_equal(vec, gold_vec):
                    wrongs.append(vec)

            alt_candidates_vectors.append(wrongs)

        if not is_sentence_valid:
            continue
        if len(gold_sequence) == 0:
            continue

        # --------------------------------------------------
        # POSITIVE SAMPLES
        # --------------------------------------------------

        for aug_idx in range(augment_factor):
            sample = pad_with_jitter(
                gold_sequence,
                window_size,
                padding_vec,
                aug_idx
            )
            if sample is not None:
                X.append(sample)
                y.append(1)
                sentence_ids.append(sent_id)

        # --------------------------------------------------
        # NEGATIVE SAMPLES
        # --------------------------------------------------

        target_negatives = neg_ratio * augment_factor
        # target_negatives = neg_ratio
        negatives_created = 0
        max_swaps = len(
            [
                i
                for i, alts in enumerate(alt_candidates_vectors)
                if len(alts) > 0
            ]
        )

        for k in range(1, max_swaps + 1):
            for neg_seq in generate_k_swap_negatives(
                gold_sequence,
                alt_candidates_vectors,
                k
            ):
                for _ in range(augment_factor):
                    sample = pad_with_jitter(
                        neg_seq,
                        window_size,
                        padding_vec,
                        np.random.randint(0, max(1, window_size - len(neg_seq) + 1))
                    )

                    if sample is None:
                        continue

                    X.append(sample)
                    y.append(0)
                    sentence_ids.append(sent_id)
                    negatives_created += 1
                    if negatives_created >= target_negatives:
                        break

            if negatives_created >= target_negatives:
                break

    return np.stack(X), np.array(y), np.array(sentence_ids)

import torch

def create_validation_groups(df, vectorize_func, window_size=40, feature_dim=64):
    val_groups = []
    padding_vec = np.zeros(feature_dim, dtype=np.float32)
    
    # Ensure selected-index is handled
    df['selected-index'] = pd.to_numeric(df['selected-index'], errors='coerce').fillna(-1).astype(int)
    
    grouped = df.groupby('sentence-id')

    for sent_id, group in grouped:
        group = group.sort_values('word-id')
        gold_sequence = []
        alt_candidates_vectors = [] # List of lists: [ [wrong_vecs_for_word1], [wrong_vecs_for_word2]... ]
        options_id = []
        is_sentence_valid = True
        
        for _, row in group.iterrows():
            try:
                # Parse the options column
                options_list = ast.literal_eval(row['options']) if isinstance(row['options'], str) else row['options']
                idx = int(row['selected-index'])
                options_id.append(options_list[idx]) # annotated index

                # Validation: must have a valid gold index
                if idx == -1 or idx >= len(options_list) or options_list[idx] == 'None of the above':
                    is_sentence_valid = False; break
                
                # 1. Vectorize and fix Gold Vector shape
                gv = np.array(vectorize_func(options_list[idx])).flatten().astype(np.float32)
                gold_sequence.append(gv)
                
                # 2. Collect all "Wrong" vectors for THIS specific word
                current_word_wrongs = []

                for i, opt in enumerate(options_list):
                    if opt == 'None of the above' or not isinstance(opt, list):
                        continue
                    wv = np.array(vectorize_func(opt)).flatten().astype(np.float32)
                    # Skip equivalent analyses
                    if np.array_equal(wv, gv):
                        continue
                    current_word_wrongs.append(wv)
                
                # We add the list (even if empty) to maintain word alignment
                alt_candidates_vectors.append(current_word_wrongs)

            except Exception as e:
                is_sentence_valid = False; break

        # Skip if sentence processing failed or gold sequence is empty
        if not is_sentence_valid or len(gold_sequence) == 0:
            continue

        # Check if there is at least one "wrong" candidate in the entire sentence
        total_negatives_count = sum(len(w) for w in alt_candidates_vectors)
        if total_negatives_count == 0:
            continue

        # --- Jitter and Batching ---
        seq_len = len(gold_sequence)
        # slack = max(0, window_size - seq_len)
        # chosen_offset = random.randint(0, slack)
        chosen_offset = 0
        # chosen_offset = slack // 2

        def apply_jitter(sequence):
            start = [padding_vec] * chosen_offset
            end = [padding_vec] * (window_size - seq_len - chosen_offset)
            return np.stack(start + sequence + end).astype(np.float32)

        # Create Gold Tensor
        gold_tensor = torch.FloatTensor(apply_jitter(gold_sequence))
        
        # Create Negative Tensors (One-word-swap logic)
        negative_tensors = []
        for i in range(len(gold_sequence)):
            # For every wrong option at word position 'i'
            for wrong_vec in alt_candidates_vectors[i]:
                neg_seq = list(gold_sequence) # copy gold
                neg_seq[i] = wrong_vec        # swap one word

                negative_tensors.append(torch.FloatTensor(apply_jitter(neg_seq)))
        
        val_groups.append({
            'sentence_id': sent_id,
            'sentence': group["sentence"].iloc[0],
            'annotated_list': options_id,
            'gold': gold_tensor,
            'negatives': negative_tensors,
        })

    return val_groups

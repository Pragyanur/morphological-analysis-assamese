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


# print(f"Shape of X: {X.shape}") # (samples, 40, 64)
# print(f"Shape of Y: {y.shape}") # (samples, 40, 64)

import matplotlib.pyplot as plt
# from matplotlib import colormaps
# print(list(colormaps))

def view_9_random_samples(X, y):
    """
    Displays a 3x3 square grid of 9 random samples from the dataset.
    Args:
        X: (samples, 40, 64) numpy array
        y: (samples,) numpy array (labels)
    """
    # 1. Select 9 random, unique indices
    num_samples = len(X)
    if num_samples < 9:
        print(f"Error: Dataset only has {num_samples} samples. Need at least 9.")
        return
        
    indices = random.sample(range(num_samples), 2)
    
    # 2. Set up the figure with 2x2 grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    
    # 3. Create a distinct color normalization to highlight sparse binary features
    ceiling=3
    norm = plt.Normalize(vmin=0, vmax=ceiling)
    
    for i, idx in enumerate(indices):
        # Calculate the grid position (row, col)
        # row_grid = i // 2
        # col_grid = i % 2
        ax = axes[i]
        
        sample = X[idx]
        label = y[idx]
        
        # Display Heatmap
        # Rows = 40 (Window), Cols = 64 (Features)
        # Using 'magma' or 'inferno' often makes sparse features pop better than viridis
        im = ax.imshow(sample, aspect='auto', cmap='Greys', norm=norm, interpolation='nearest')
        # 4. Title and labels
        label_text = 'GOLD (1)' if label == 1 else 'WRONG (0)'
        ax.set_title(f"Idx: {idx} | {label_text}")

        ax.set_xlabel("Features (64)")
        # Labels only for the outer edge of the grid
        if i == 0:
            ax.set_ylabel("Context (40)")

    # 5. Add a single colorbar for the whole figure
    # This colorbar is now relative to the 3x3 grid, not just one plot
    # plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.01, pad=0.02, label=f'Feature Weight (0-{ceiling})')
    plt.show()

# def view_gold_wrong_pair(X, y, sentence_ids):
#     """
#     Displays one GOLD sample and one WRONG sample
#     from the same sentence-id side by side.
#     """

#     # Find sentence ids having both gold and wrong samples
#     valid_sentence_ids = []

#     for sid in np.unique(sentence_ids):
#         labels = y[sentence_ids == sid]

#         if 1 in labels and 0 in labels:
#             valid_sentence_ids.append(sid)

#     if not valid_sentence_ids:
#         print("No sentence-id contains both GOLD and WRONG samples.")
#         return

#     # Randomly choose one sentence-id
#     chosen_sid = random.choice(valid_sentence_ids)

#     # Get indices
#     gold_indices = np.where((sentence_ids == chosen_sid) & (y == 1))[0]
#     wrong_indices = np.where((sentence_ids == chosen_sid) & (y == 0))[0]

#     gold_idx = random.choice(gold_indices)
#     wrong_idx = random.choice(wrong_indices)

#     indices = [gold_idx, wrong_idx]
#     titles = ["GOLD (1)", "WRONG (0)"]

#     # Plot
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

#     ceiling = 3
#     norm = plt.Normalize(vmin=0, vmax=ceiling)

#     for i, idx in enumerate(indices):

#         ax = axes[i]

#         im = ax.imshow(
#             X[idx],
#             aspect='auto',
#             cmap='coolwarm',
#             norm=norm,
#             interpolation='nearest'
#         )

#         ax.set_title(f"Sentence: {chosen_sid} | {titles[i]}")
#         ax.set_xlabel("Features (64)")

#         if i == 0:
#             ax.set_ylabel("Context (64)")

#     fig.colorbar(
#         im,
#         ax=axes.ravel().tolist(),
#         fraction=0.01,
#         pad=0.02,
#         label=f'Feature Weight (0-{ceiling})'
#     )

#     plt.show()

def view_gold_wrong_pair(X, y, sentence_ids):
    """
    Displays one GOLD sample and one WRONG sample
    from the same sentence-id side by side.

    Zero-vector rows in X are removed before plotting.
    """

    # Find sentence ids having both gold and wrong samples
    valid_sentence_ids = []

    for sid in np.unique(sentence_ids):
        labels = y[sentence_ids == sid]

        if 1 in labels and 0 in labels:
            valid_sentence_ids.append(sid)

    if not valid_sentence_ids:
        print("No sentence-id contains both GOLD and WRONG samples.")
        return

    # Randomly choose one sentence-id
    chosen_sid = random.choice(valid_sentence_ids)

    # Get indices
    gold_indices = np.where(
        (sentence_ids == chosen_sid) & (y == 1)
    )[0]

    wrong_indices = np.where(
        (sentence_ids == chosen_sid) & (y == 0)
    )[0]

    gold_idx = random.choice(gold_indices)
    wrong_idx = random.choice(wrong_indices)

    indices = [gold_idx, wrong_idx]
    titles = ["GOLD (1)", "WRONG (0)"]

    # Plot
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4),
        sharex=True,
        sharey=True
    )

    ceiling = 2.5
    norm = plt.Normalize(vmin=0, vmax=ceiling)

    for i, idx in enumerate(indices):

        ax = axes[i]

        # --------------------------------------------------
        # Remove rows that are completely zero
        # --------------------------------------------------
        zero_rows = np.all(X[idx] == 0, axis=1)
        X_seq = X[idx][~zero_rows]

        # Actual sequence length after removing zero vectors
        seq_len = X_seq.shape[0]

        im = ax.imshow(
            X_seq,
            aspect='auto',
            cmap='Grays',
            norm=norm,
            interpolation='nearest'
        )

        ax.set_xticks(
            np.arange(-0.5, X_seq.shape[1], 1),
            minor=True
        )
        
        ax.set_yticks(
            np.arange(-0.5, X_seq.shape[0], 1),
            minor=True
        )

        ax.grid(
            which='minor',
            linewidth=0.5
        )

        # Don't show minor tick marks
        ax.tick_params(
            which='minor',
            bottom=False,
            left=False
        )

        ax.set_title(
            f"Sentence: {chosen_sid} | {titles[i]} | "
            f"Length: {seq_len}"
        )

        ax.set_xlabel("Feature indices (64)")

        if i == 0:
            ax.set_ylabel("Sequence position")

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        fraction=0.01,
        pad=0.02,
        label=f'Feature Weight (0-{ceiling})'
    )

    plt.show()

# X,y,sentence_indices = create_complete_morphology_dataset(pd.read_csv("annotation/annotated-dataset/1072-annotated-dataset-updated.csv"), vectorize_morphology)
# view_gold_wrong_pair(X,y,sentence_indices)



# previous version that only consider 1 swap negatives
# def create_complete_morphology_dataset(
#     df, 
#     vectorize_func=vectorize_morphology, 
#     window_size=40, 
#     feature_dim=64, 
#     neg_ratio=3, 
#     augment_factor=3
# ):
#     X = []
#     y = []
#     sentence_ids = []
#     padding_vec = np.zeros(feature_dim, dtype=np.float32)
    
#     df['selected-index'] = df['selected-index'].fillna(-1).astype(int)
#     grouped = df.groupby('sentence-id')
    
#     for sent_id, group in grouped:
#         group = group.sort_values('word-id')
#         gold_sequence = []
#         alt_candidates_vectors = []
#         is_sentence_valid = True
        
#         for _, row in group.iterrows():
#             try:
#                 options_list = ast.literal_eval(row['options'])
#             except:
#                 is_sentence_valid = False; break
                
#             idx = row['selected-index']
            
#             # Skip sentence if ground truth is "None of the above"
#             if idx == -1 or idx >= len(options_list) or options_list[idx] == 'None of the above':
#                 is_sentence_valid = False; break
            
#             # REPLACED WITH
#             # --- Gold vector ---
#             gold_vec = np.array(vectorize_func(options_list[idx])).flatten().astype(np.float32)

#             if gold_vec.shape[0] != feature_dim:
#                 continue

#             gold_sequence.append(gold_vec)

#             # --- Find equivalent analyses ---
#             valid_vectors = []
#             wrongs = []

#             for i, opt in enumerate(options_list):
#                 if opt == 'None of the above' or not isinstance(opt, list):
#                     continue
#                 vec = np.array(vectorize_func(opt)).flatten().astype(np.float32)
#                 if vec.shape[0] != feature_dim:
#                     continue
#                 # Treat identical vectors as equivalent gold analyses
#                 if np.array_equal(vec, gold_vec):
#                     valid_vectors.append(vec)
#                 else:
#                     wrongs.append(vec)

#             alt_candidates_vectors.append(wrongs)

#         if not is_sentence_valid or not gold_sequence:
#             continue

#         # Add Positive Samples
#         for _ in range(augment_factor):
#             sample = pad_with_jitter(gold_sequence, window_size, padding_vec, feature_dim)
#             X.append(sample)
#             y.append(1)
#             sentence_ids.append(sent_id)
        
#         # Add Negative Samples
#         changeable_indices = [i for i, v in enumerate(alt_candidates_vectors) if v]
#         if changeable_indices:
#             for _ in range(neg_ratio * augment_factor):
#                 neg_seq = list(gold_sequence)
#                 num_to_swap = min(len(changeable_indices), random.randint(1, 2)) # one swap only
#                 for tidx in random.sample(changeable_indices, num_to_swap):
#                     neg_seq[tidx] = random.choice(alt_candidates_vectors[tidx])
                
#                 sample = pad_with_jitter(neg_seq, window_size, padding_vec, feature_dim)
#                 X.append(sample)
#                 y.append(0)
#                 sentence_ids.append(sent_id)

#     # Final conversion to a single 3D block
#     return np.stack(X), np.array(y), np.array(sentence_ids)


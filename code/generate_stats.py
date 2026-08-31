import ast
import numpy as np
import pandas as pd


def extract_dataset_statistics(csv_path):
    print(f"Loading dataset from: {csv_path}...")

    try:
        df = pd.read_csv(csv_path)

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # ---------------------------------------------------------
    # Check required columns
    # ---------------------------------------------------------

    required_columns = [
        "sentence-id",
        "word-id",
        "sentence",
        "word",
        "options"
    ]

    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in CSV.")
            return

    print(
        "Analyzing dataset structure. Please wait...\n"
        + "=" * 60
    )

    # =========================================================
    # 1. Basic Token and Word Statistics
    # =========================================================

    total_tokens = len(df)

    total_unique_words = (
        df["word"]
        .astype(str)
        .str.lower()
        .nunique()
    )

    # =========================================================
    # 2. Sentence-level calculations
    # =========================================================

    grouped_sentences = df.groupby("sentence-id")

    total_sentences = len(grouped_sentences)

    sentence_lengths = []

    for sent_id, group in grouped_sentences:
        sentence_lengths.append(len(group))

    sentence_lengths = np.array(sentence_lengths)

    avg_sentence_length = sentence_lengths.mean()
    max_sentence_length = sentence_lengths.max()
    min_sentence_length = sentence_lengths.min()

    # =========================================================
    # 3. Morphological options & ambiguity calculations
    # =========================================================

    all_option_counts = []
    ambiguous_tokens_count = 0

    # Maximum number of analyses for any single word
    max_word_analyses = 0
    max_word_analyses_word = None
    max_word_analyses_sentence_id = None

    # Store number of candidate sequences for every sentence
    sentence_candidate_counts = []

    for sent_id, group in grouped_sentences:

        num_candidates = 1

        for _, row in group.iterrows():

            options_raw = row["options"]

            try:
                if isinstance(options_raw, str):
                    options_list = ast.literal_eval(options_raw)
                else:
                    options_list = options_raw

            except Exception:
                options_list = []

            if not isinstance(options_list, list):
                options_list = []

            cleaned_options = [
                opt
                for opt in options_list
                if opt != "None of the above"
            ]

            num_options = len(cleaned_options)

            # Token-level statistics
            all_option_counts.append(num_options)

            if num_options > 1:
                ambiguous_tokens_count += 1

            # Maximum analyses for any word
            if num_options > max_word_analyses:
                max_word_analyses = num_options
                max_word_analyses_word = row["word"]
                max_word_analyses_sentence_id = sent_id

            # Sentence-level candidate count
            num_candidates *= num_options

        sentence_candidate_counts.append(num_candidates)

    # =========================================================
    # 4. Average analyses per token
    # =========================================================

    avg_analyses_options = (
        np.mean(all_option_counts)
        if all_option_counts
        else 0.0
    )

    # =========================================================
    # 5. Candidate sequence statistics
    # =========================================================

    if sentence_candidate_counts:

        max_candidate_sequences = max(
            sentence_candidate_counts
        )

        median_candidate_sequences = float(
            np.median(
                np.array(
                    sentence_candidate_counts,
                    dtype=object
                )
            )
        )

    else:
        max_candidate_sequences = 0
        median_candidate_sequences = 0

    # =========================================================
    # 6. Print report
    # =========================================================

    print("📊 DATASET STATISTICS REPORT")
    print("-" * 60)

    print(
        f"• Total Sentences:                    "
        f"{total_sentences:,}"
    )

    print(
        f"• Total Token Instances (Rows):       "
        f"{total_tokens:,}"
    )

    print(
        f"• Total Unique Vocabulary Words:      "
        f"{total_unique_words:,}"
    )

    print("-" * 60)

    print(
        f"• Ambiguous Tokens (>1 option):        "
        f"{ambiguous_tokens_count:,} "
        f"({(ambiguous_tokens_count / total_tokens) * 100:.2f}% of text)"
    )

    print(
        f"• Avg. Analyses Options per Token:    "
        f"{avg_analyses_options:.2f} "
        f"(excl. 'None of the above')"
    )

    print("-" * 60)

    print(
        f"• Average Sentence Length:            "
        f"{avg_sentence_length:.2f} tokens"
    )

    print(
        f"• Maximum Sentence Length:             "
        f"{max_sentence_length} tokens"
    )

    print(
        f"• Minimum Sentence Length:             "
        f"{min_sentence_length} tokens"
    )

    print("-" * 60)

    print(
        f"• Median Candidate Sequences/Sentence:"
        f" {median_candidate_sequences:.4e}"
    )

    print(
        f"• Maximum Candidate Sequences/Sentence:"
        f" {max_candidate_sequences:.4e}"
    )

    print("-" * 60)

    print(
        f"• Maximum Analyses for Any Word:       "
        f"{max_word_analyses}"
    )

    print(
        f"  Word:                                "
        f"{max_word_analyses_word}"
    )

    print(
        f"  Sentence ID:                         "
        f"{max_word_analyses_sentence_id}"
    )

    print("=" * 60)

# =============================================================
# Run
# =============================================================

if __name__ == "__main__":

    csv_file_path = (
        "dataset/1072-annotated-dataset-updated.csv"
    )

    extract_dataset_statistics(csv_file_path)
import pandas as pd
import ast
import glob
import os

INPUT_FILES = ["annotation/annotated-dataset/combined_misalignment-free.csv"]
OUTPUT_FILE = "annotation/annotated-dataset/combined_misalignment-free_problemetic-rows.csv"

def problem_csv_creation(df, output_file=OUTPUT_FILE):

    df['selected-index'] = df['selected-index'].fillna(-1).astype(int)

    problematic_rows = []
    for idx, row in df.iterrows():
        try:
            options_list = ast.literal_eval(row['options'])
        except Exception:
            problematic_rows.append(row)
            continue

        selected_idx = row['selected-index']

        needs_manual = (
            selected_idx < 0
            or selected_idx >= len(options_list)
            or options_list[selected_idx] == 'None of the above'
        )

        if needs_manual:
            problematic_rows.append(row)

    problem_df = pd.DataFrame(problematic_rows)
    problem_df.to_csv(output_file, index=False)
    print(f"Saved {len(problem_df)} problematic rows to {output_file}")


# data = pd.read_csv("annotation/annotated-dataset/combined_misalignment-free.csv")
# problem_csv_creation(data, "annotation/annotated-dataset/combined_problematic-rows.csv")

def regenerate_options_preserve_selection(
    input_csv,
    output_csv,
    generator_func
):
    """
    Regenerates options for every row using generator_func(word),
    while preserving the previously selected analysis whenever possible.

    Strategy:
    - Previously selected option is preserved.
    - Newly generated options are appended before
      'None of the above'.
    - Duplicate options are removed.
    - selected-index is updated automatically.

    Parameters
    ----------
    input_csv : str
        Original annotation CSV

    output_csv : str
        Updated output CSV

    generator_func : callable
        Function:
            generator_func(word) -> list of analyses
    """

    df = pd.read_csv(input_csv)

    updated_rows = []

    for _, row in df.iterrows():

        try:
            old_options = ast.literal_eval(row['options'])
        except Exception:
            updated_rows.append(row)
            continue

        selected_idx = int(row['selected-index']) \
            if pd.notna(row['selected-index']) else -1

        # -----------------------------------------------------
        # Preserve previously selected analysis
        # -----------------------------------------------------
        selected_option = None

        if (
            0 <= selected_idx < len(old_options)
            and old_options[selected_idx] != 'None of the above'
        ):
            selected_option = old_options[selected_idx]

        # -----------------------------------------------------
        # Generate NEW options
        # -----------------------------------------------------
        try:
            new_generated = generator_func(row['word'])

        except Exception:
            updated_rows.append(row)
            continue

        # Ensure list
        if not isinstance(new_generated, list):
            new_generated = []

        # -----------------------------------------------------
        # Merge while removing duplicates
        # -----------------------------------------------------
        merged = []

        # Preserve selected first
        if selected_option is not None:
            merged.append(selected_option)

        # Add newly generated options
        for opt in new_generated:
            if opt not in merged:
                merged.append(opt)

        # Add NONE option at end
        merged.append('None of the above')

        # -----------------------------------------------------
        # Update selected-index
        # -----------------------------------------------------
        new_selected_idx = -1

        if selected_option is not None:
            try:
                new_selected_idx = merged.index(selected_option)
            except ValueError:
                new_selected_idx = -1

        # -----------------------------------------------------
        # Update row
        # -----------------------------------------------------
        row['options'] = str(merged)
        row['selected-index'] = new_selected_idx

        updated_rows.append(row)

    updated_df = pd.DataFrame(updated_rows)

    updated_df.to_csv(output_csv, index=False)

    print(f"Saved updated dataset to: {output_csv}")

import pandas as pd
import ast
from morphology_inrule import rule_based_forward_analysis


# =========================================================
# INPUT FILES
# =========================================================
FILE_A = "annotation/annotated-dataset/problematic-rows-mark.csv"      # sentence-id, sentence-new
FILE_B = "annotation/annotated-dataset/combined_misalignment-free.csv"    # sentence-id, word-id, sentence, word, options, selected-index

# =========================================================
# OUTPUT FILES
# =========================================================
UPDATED_B = "annotation/updated_annotations.csv"
NEW_WORDS = "annotation/last_batch.csv"

# =========================================================
# LOAD CSVs
# =========================================================
df_a = pd.read_csv(FILE_A)
df_b = pd.read_csv(FILE_B)

# Normalize column names if needed
df_a.columns = [c.strip() for c in df_a.columns]
df_b.columns = [c.strip() for c in df_b.columns]

# =========================================================
# LOOKUP
# =========================================================
sentence_lookup = dict(zip(df_a["sentence-id"], df_a["sentence-new"]))

# =========================================================
# PROCESS
# =========================================================
updated_rows = []
new_rows = []

grouped = df_b.groupby("sentence-id")

for sentence_id, group in grouped:

    if sentence_id not in sentence_lookup:
        continue

    new_sentence = str(sentence_lookup[sentence_id]).strip()
    new_tokens = new_sentence.split()

    # Store existing rows by word
    existing_rows = {}

    for _, row in group.iterrows():
        word = str(row["word"]).strip()

        if word not in existing_rows:
            existing_rows[word] = row.to_dict()

    # Reinitialize word-id
    new_word_id = 0

    for token in new_tokens:

        # =====================================================
        # EXISTING TOKEN -> KEEP OLD DATA
        # =====================================================
        if token in existing_rows:

            old_row = existing_rows[token]

            updated_row = {
                "sentence-id": sentence_id,
                "word-id": new_word_id,
                "sentence": new_sentence,
                "word": token,
                "options": old_row.get("options", ""),
                "selected-index": old_row.get("selected-index", "")
            }

            updated_rows.append(updated_row)

        # =====================================================
        # NEW TOKEN -> GENERATE OPTIONS
        # =====================================================
        else:

            generated_options = rule_based_forward_analysis(
                word=token
            )

            new_row = {
                "sentence-id": sentence_id,
                "word-id": new_word_id,
                "sentence": new_sentence,
                "word": token,
                "options": generated_options,
                "selected-index": ""
            }

            new_rows.append(new_row)

        new_word_id += 1

# =========================================================
# SAVE
# =========================================================
updated_df = pd.DataFrame(updated_rows)
new_words_df = pd.DataFrame(new_rows)

new_words_df = new_words_df.drop(columns=["selected-index"])

updated_df.to_csv(UPDATED_B, index=False, encoding="utf-8")
new_words_df.to_csv(NEW_WORDS, index=False, encoding="utf-8")

print(f"Saved updated annotations to: {UPDATED_B}")
print(f"Saved new words with generated options to: {NEW_WORDS}")













# import pandas as pd

# # =========================================================
# # FILES
# # =========================================================
# # FILE_A = "sentences.csv"              # sentence-id, sentence-new
# FILE_C = "annotation/annotated-dataset/combined_problematic-rows.csv"     # same structure as B

# OUTPUT_FILE = "annotation/reinitialized_options.csv"

# # =========================================================
# # LOAD
# # =========================================================
# df_a = pd.read_csv(FILE_A)
# df_c = pd.read_csv(FILE_C)

# df_a.columns = [c.strip() for c in df_a.columns]
# df_c.columns = [c.strip() for c in df_c.columns]

# # =========================================================
# # REMOVE ROWS HAVING sentence-id IN A
# # =========================================================
# sentence_ids_in_a = set(df_a["sentence-id"])

# df_c = df_c[~df_c["sentence-id"].isin(sentence_ids_in_a)].copy()

# # =========================================================
# # REGENERATE OPTIONS ONLY
# # =========================================================
# new_options = []

# for idx, row in df_c.iterrows():

#     word = str(row["word"]).strip()
#     sentence = str(row["sentence"]).strip()

#     generated_options = rule_based_forward_analysis(
#         word=word
#     )

#     new_options.append(generated_options)

# # Replace ONLY options column
# df_c["options"] = new_options

# # =========================================================
# # SAVE
# # =========================================================
# df_c = df_c.drop(columns=["selected-index"])

# df_c.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# print(f"Saved regenerated file to: {OUTPUT_FILE}")
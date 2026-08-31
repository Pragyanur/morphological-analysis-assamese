import pandas as pd
import ast

# Input/output files
input_file = "dataset/1072-annotated-dataset-updated.csv"
output_file = "dataset/1072-annotated-dataset-updated.csv"

df = pd.read_csv(input_file)

# =========================================================
# 1. Make selected-index consistently float
# =========================================================

df["selected-index"] = pd.to_numeric(
    df["selected-index"],
    errors="coerce"
).astype(float)


# =========================================================
# 2. Identify sentences containing "none of the above"
#    that was selected
# =========================================================

sentences_to_remove = set()

for _, row in df.iterrows():

    selected_index = row["selected-index"]

    # Skip rows where selected-index is missing
    if pd.isna(selected_index):
        continue

    try:
        options = ast.literal_eval(row["options"])
    except (ValueError, SyntaxError):
        print(f"Could not parse options: {row['options']}")
        continue

    # Find "none of the above"
    none_index = None

    for idx, option in enumerate(options):
        if str(option).strip().lower() == "none of the above":
            none_index = idx
            break

    # If this row selected "none of the above",
    # mark the entire sentence for removal.
    if none_index is not None and selected_index == float(none_index):
        sentences_to_remove.add(row["sentence-id"])


# =========================================================
# 3. Remove entire sentence groups
# =========================================================

before_sentences = df["sentence-id"].nunique()

df = df[
    ~df["sentence-id"].isin(sentences_to_remove)
].copy()

after_sentences = df["sentence-id"].nunique()


# =========================================================
# 4. Remove "none of the above" from remaining options
# =========================================================

def remove_none_of_the_above(options_string):
    try:
        options = ast.literal_eval(options_string)

        options = [
            option
            for option in options
            if str(option).strip().lower() != "none of the above"
        ]

        return str(options)

    except (ValueError, SyntaxError):
        print(f"Could not parse options: {options_string}")
        return options_string


df["options"] = df["options"].apply(remove_none_of_the_above)


# =========================================================
# 5. Re-enumerate sentence IDs
# =========================================================

new_sentence_ids = []

current_id = -1
previous_sentence = None

for sentence in df["sentence"]:

    if sentence != previous_sentence:
        current_id += 1
        previous_sentence = sentence

    new_sentence_ids.append(current_id)

df["sentence-id"] = new_sentence_ids


# =========================================================
# 6. Save
# =========================================================

df.to_csv(output_file, index=False)

print(f"Removed {before_sentences - after_sentences} sentences.")
print(f"Removed 'none of the above' from remaining options.")
print(f"Done. {current_id + 1} sentences written to {output_file}")
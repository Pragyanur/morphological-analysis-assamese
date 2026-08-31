import csv
import ast
import morphology_inrule as mir


# =========================================================
# Files
# =========================================================

INPUT_FILE = "annotation/annotated-dataset/1072-sentences-noNota.csv"
OUTPUT_FILE = "annotation/annotated-dataset/1072-sentences-noNota-updated.csv"

NONE_OPTION = "None of the above"


# =========================================================
# Read CSV
# =========================================================

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)


print("Columns found:")
print(reader.fieldnames)
print(f"Total rows: {len(rows)}")


# =========================================================
# Statistics
# =========================================================

updated_rows = []

total = 0
preserved = 0
changed_to_none = 0
errors = 0


# =========================================================
# Process rows
# =========================================================

for row in rows:

    total += 1

    sentence_id = row["sentence-id"]
    word_id = int(float(row["word-id"]))
    sentence = row["sentence"]
    old_word = row["word"]

    # -----------------------------------------------------
    # Parse OPTIONS
    # -----------------------------------------------------

    options_string = row["options"]

    try:
        old_options = ast.literal_eval(options_string)

        if not isinstance(old_options, list):
            raise ValueError(
                f"options is not a list: {type(old_options)}"
            )

    except Exception as e:

        print("\nERROR READING OPTIONS")
        print(f"sentence-id : {sentence_id}")
        print(f"word-id     : {word_id}")
        print(f"word        : {old_word!r}")
        print(f"raw options : {options_string!r}")
        print(f"error       : {e}")

        errors += 1

        # Preserve original row
        updated_rows.append(row)
        continue


    # -----------------------------------------------------
    # Parse SELECTED INDEX
    #
    # CSV contains values such as:
    #     0.0
    #     1.0
    #     2.0
    #
    # -----------------------------------------------------

    try:
        selected_index = int(float(row["selected-index"]))

    except Exception as e:

        print("\nERROR READING SELECTED INDEX")
        print(f"sentence-id   : {sentence_id}")
        print(f"word-id       : {word_id}")
        print(f"selected-index: {row['selected-index']!r}")
        print(f"error         : {e}")

        errors += 1
        updated_rows.append(row)
        continue


    # -----------------------------------------------------
    # Validate selected index
    # -----------------------------------------------------

    if selected_index < 0 or selected_index >= len(old_options):

        print("\nINVALID SELECTED INDEX")
        print(f"sentence-id   : {sentence_id}")
        print(f"word-id       : {word_id}")
        print(f"selected-index: {selected_index}")
        print(f"number options: {len(old_options)}")
        print(f"options       : {old_options}")

        errors += 1
        updated_rows.append(row)
        continue


    # -----------------------------------------------------
    # Recover ACTUAL selected option
    # -----------------------------------------------------

    old_selected_option = old_options[selected_index]


    # =====================================================
    # Generate NEW OPTIONS
    # =====================================================

    try:
        new_sent_options = mir.sentence_word_options(sentence)

    except Exception as e:

        print("\nERROR GENERATING NEW OPTIONS")
        print(f"sentence-id: {sentence_id}")
        print(f"sentence   : {sentence!r}")
        print(f"error      : {e}")

        errors += 1
        updated_rows.append(row)
        continue


    # -----------------------------------------------------
    # Validate word ID
    # -----------------------------------------------------

    if word_id >= len(new_sent_options):

        print("\nWORD ID OUT OF RANGE")
        print(f"sentence-id       : {sentence_id}")
        print(f"word-id           : {word_id}")
        print(f"new options length: {len(new_sent_options)}")

        errors += 1
        updated_rows.append(row)
        continue


    # -----------------------------------------------------
    # Get NEW word and options
    # -----------------------------------------------------

    new_word, new_options = new_sent_options[word_id]

    new_options = list(new_options)


    # -----------------------------------------------------
    # Verify word correspondence
    # -----------------------------------------------------

    if new_word != old_word:

        print("\nWORD MISMATCH")
        print(f"sentence-id: {sentence_id}")
        print(f"word-id    : {word_id}")
        print(f"OLD word   : {old_word!r}")
        print(f"NEW word   : {new_word!r}")

        errors += 1
        updated_rows.append(row)
        continue


    # =====================================================
    # Add "None of the above"
    # =====================================================

    if NONE_OPTION in new_options:

        none_index = new_options.index(NONE_OPTION)

    else:

        new_options.append(NONE_OPTION)
        none_index = len(new_options) - 1


    # =====================================================
    # Preserve selection
    # =====================================================

    if old_selected_option in new_options:

        new_selected_index = new_options.index(
            old_selected_option
        )

        preserved += 1

    else:

        new_selected_index = none_index

        changed_to_none += 1

        print(
            f"\nSelection no longer available:"
            f"\n  sentence-id : {sentence_id}"
            f"\n  word-id     : {word_id}"
            f"\n  word        : {old_word!r}"
            f"\n  old selection: {old_selected_option!r}"
            f"\n  → None of the above"
        )


    # =====================================================
    # Update row
    # =====================================================

    row["word"] = new_word

    # Store list as a string
    row["options"] = repr(new_options)

    # Store index as float string, matching original format
    row["selected-index"] = str(float(new_selected_index))

    updated_rows.append(row)


# =========================================================
# Write UPDATED CSV
# =========================================================

fieldnames = [
    "sentence-id",
    "word-id",
    "sentence",
    "word",
    "options",
    "selected-index"
]


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(updated_rows)


# =========================================================
# SUMMARY
# =========================================================

print("\n")
print("=" * 60)
print("UPDATE COMPLETE")
print("=" * 60)

print(f"Total rows              : {total}")
print(f"Selections preserved    : {preserved}")
print(f"Changed to None         : {changed_to_none}")
print(f"Errors                  : {errors}")
print(f"Output                  : {OUTPUT_FILE}")

print("=" * 60)

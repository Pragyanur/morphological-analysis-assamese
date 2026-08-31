# import json
# import pandas as pd
# import ast
# from collections import Counter


# # =========================================================
# # Input files
# # =========================================================

# input_file = "annotation/annotated-dataset/1071-sentences-options-updated-noNota-x.csv"
# tags_file = "data/tag-seq-rules-updated.json"

# output_file = "annotation/empirical-bigram-probabilities.csv"


# # =========================================================
# # Load annotated data
# # =========================================================

# df = pd.read_csv(input_file)


# # =========================================================
# # Extract selected options
# # =========================================================

# selected_options = []

# for _, row in df.iterrows():

#     options = ast.literal_eval(row["options"])

#     selected_index = int(float(row["selected-index"]))

#     selected_option = options[selected_index]

#     selected_options.append(selected_option)


# # =========================================================
# # Construct tag sequences
# # =========================================================

# sequences = []

# for t_seq in selected_options:

#     seq = ["ws"]

#     for t in t_seq:
#         seq.append(t[1])

#     seq.append("we")

#     sequences.append(seq)


# # =========================================================
# # Load tag dictionary
# # =========================================================

# with open(tags_file, "r", encoding="utf-8") as f:
#     tags = json.load(f)


# # =========================================================
# # Get tag inventory
# # =========================================================

# # Assuming the JSON dictionary has the tags as keys.
# tag_inventory = list(tags.keys())

# # Add sentence boundary tags if they are not already present.
# if "ws" not in tag_inventory:
#     tag_inventory.append("ws")

# if "we" not in tag_inventory:
#     tag_inventory.append("we")


# print(f"Number of tags: {len(tag_inventory)}")


# # =========================================================
# # Count bigrams
# # =========================================================

# bigram_counts = Counter()

# # Number of times each tag occurs as the first element
# # of a bigram.
# first_tag_counts = Counter()


# for seq in sequences:

#     for i in range(len(seq) - 1):

#         first = seq[i]
#         second = seq[i + 1]

#         bigram_counts[(first, second)] += 1
#         first_tag_counts[first] += 1


# # =========================================================
# # Calculate empirical probabilities
# # =========================================================

# results = []

# for first in tag_inventory:

#     denominator = first_tag_counts[first]

#     for second in tag_inventory:

#         count = bigram_counts[(first, second)]

#         if denominator > 0:
#             probability = count / denominator
#         else:
#             probability = 0.0

#         results.append({
#             "tag1": first,
#             "tag2": second,
#             # "bigram": f"{first} {second}",
#             # "count": count,
#             # "tag1_count": denominator,
#             "probability": probability
#         })


# # =========================================================
# # Save
# # =========================================================

# bigram_df = pd.DataFrame(results)

# bigram_df.to_csv(
#     output_file,
#     index=False
# )


# print(f"Saved bigram probabilities to: {output_file}")
# print(f"Number of bigrams: {len(bigram_df)}")







# import json
# import pandas as pd
# import ast
# from collections import Counter, defaultdict

# # =========================================================
# # Input files
# # =========================================================

# input_file = "annotation/annotated-dataset/1072-annotated-dataset.csv"
# tags_file = "data/tag-seq-rules-updated.json"

# # =========================================================
# # Load annotated data
# # =========================================================

# df = pd.read_csv(input_file)


# # =========================================================
# # Load tag dictionary
# # =========================================================

# with open(tags_file, "r", encoding="utf-8") as f:
#     tags = json.load(f)

# # =========================================================
# # Extract selected options
# # =========================================================

# selected_options = []

# for _, row in df.iterrows():
#     options = ast.literal_eval(row["options"])
#     selected_index = int(float(row["selected-index"]))
#     selected_option = options[selected_index]
#     selected_options.append(selected_option)

# # =========================================================
# # Construct tag sequences
# # =========================================================

# sequences = []

# for t_seq in selected_options:
#     subseq = []
#     for t in t_seq:
#         subseq.append(t[1])
#     if subseq[0] not in tags.keys():
#         continue
#     else:
#         # seq = ["ws"]
#         # seq.extend(subseq)
#         # seq.append("we")
#         sequences.append(subseq)

# # =========================================================
# # Get tag inventory
# # =========================================================

# tag_inventory = list(tags.keys())

# if "ws" not in tag_inventory:
#     tag_inventory.append("ws")

# if "we" not in tag_inventory:
#     tag_inventory.append("we")

# print(f"Number of tags: {len(tag_inventory)}")

# # =========================================================
# # Count bigrams
# # =========================================================

# bigram_counts = Counter()
# first_tag_counts = Counter()

# for seq in sequences:
#     for i in range(len(seq) - 1):
#         first = seq[i]
#         second = seq[i + 1]
#         bigram_counts[(first, second)] += 1
#         first_tag_counts[first] += 1

# # =========================================================
# # Create transition probability matrix with add-one smoothing
# # =========================================================

# num_tags = len(tag_inventory)

# tag_to_index = {
#     tag: index
#     for index, tag in enumerate(tag_inventory)
# }

# # Start every transition with count = 1
# # This is Laplace/add-one smoothing.
# transition_counts = [
#     [1] * num_tags
#     for _ in range(num_tags)
# ]

# # Add observed bigram counts
# for (first, second), count in bigram_counts.items():

#     i = tag_to_index[first]
#     j = tag_to_index[second]

#     transition_counts[i][j] += count


# # =========================================================
# # Normalize each row
# # =========================================================

# transition_matrix = [
#     [0.0] * num_tags
#     for _ in range(num_tags)
# ]

# for first in tag_inventory:

#     i = tag_to_index[first]

#     # Number of times this tag occurred as the first
#     # element of a bigram
#     denominator = first_tag_counts[first] + num_tags

#     for j in range(num_tags):

#         transition_matrix[i][j] = (
#             transition_counts[i][j] / denominator
#         )


# # =========================================================
# # Verify probabilities
# # =========================================================

# for i, tag in enumerate(tag_inventory):

#     row_sum = sum(transition_matrix[i])

#     print(
#         f"{tag}: row sum = {row_sum:.6f}"
#     )

# # =========================================================
# # Convert to DataFrame and save
# # =========================================================

# bigram_df = pd.DataFrame(transition_matrix, columns=tag_inventory, index=tag_inventory)

# print(bigram_df)

# # If you want to save the matrix as a CSV file, uncomment the following lines:
# output_file = "annotation/empirical-bigram-probabilities.csv"
# bigram_df.to_csv(output_file)
# print(f"Saved bigram probabilities to: {output_file}")




import json
import pandas as pd
import ast
from collections import Counter

# =========================================================
# Input files
# =========================================================

input_file = "annotation/annotated-dataset/1072-annotated-dataset.csv"
tags_file = "data/tag-seq-rules-updated.json"

# =========================================================
# Output files
# =========================================================

count_output_file = "annotation/empirical-bigram-counts.csv"
probability_output_file = "annotation/empirical-bigram-probabilities.csv"

# =========================================================
# Load annotated data
# =========================================================

df = pd.read_csv(input_file)

# =========================================================
# Load tag dictionary
# =========================================================

with open(tags_file, "r", encoding="utf-8") as f:
    tags = json.load(f)

# =========================================================
# Extract selected options
# =========================================================

selected_options = []

for _, row in df.iterrows():

    options = ast.literal_eval(row["options"])
    selected_index = int(float(row["selected-index"]))

    selected_option = options[selected_index]

    selected_options.append(selected_option)

# =========================================================
# Construct tag sequences
# =========================================================

sequences = []

for t_seq in selected_options:

    subseq = []

    for t in t_seq:
        subseq.append(t[1])

    # Ignore sequences whose first tag is not
    # present in the tag dictionary
    if len(subseq) == 0:
        continue

    if subseq[0] not in tags.keys():
        continue

    sequences.append(subseq)

# =========================================================
# Get tag inventory
# =========================================================

tag_inventory = list(tags.keys())

if "ws" not in tag_inventory:
    tag_inventory.append("ws")

if "we" not in tag_inventory:
    tag_inventory.append("we")

print(f"Number of tags: {len(tag_inventory)}")

# =========================================================
# Create tag-to-index mapping
# =========================================================

num_tags = len(tag_inventory)

tag_to_index = {
    tag: index
    for index, tag in enumerate(tag_inventory)
}

# =========================================================
# Count bigrams
# =========================================================

bigram_counts = Counter()

# Count how many times each tag occurs as the
# first element of a bigram
first_tag_counts = Counter()

for seq in sequences:

    for i in range(len(seq) - 1):

        first = seq[i]
        second = seq[i + 1]

        bigram_counts[(first, second)] += 1
        first_tag_counts[first] += 1

# =========================================================
# Create UNSMOOTHED bigram count matrix
# =========================================================

count_matrix = [
    [0] * num_tags
    for _ in range(num_tags)
]

for (first, second), count in bigram_counts.items():

    # Skip tags that are not in the inventory
    if first not in tag_to_index or second not in tag_to_index:
        continue

    i = tag_to_index[first]
    j = tag_to_index[second]

    count_matrix[i][j] = count

# =========================================================
# Convert count matrix to DataFrame
# =========================================================

bigram_count_df = pd.DataFrame(
    count_matrix,
    columns=tag_inventory,
    index=tag_inventory
)

# =========================================================
# Print unsmoothed counts
# =========================================================

print("\n==============================================")
print("UNSMOOTHED BIGRAM COUNTS")
print("==============================================")

print(bigram_count_df)

# =========================================================
# Save unsmoothed counts
# =========================================================

bigram_count_df.to_csv(count_output_file)

print(
    f"\nSaved unsmoothed bigram counts to: "
    f"{count_output_file}"
)

# =========================================================
# Create transition probability matrix
# with add-one / Laplace smoothing
# =========================================================

# Start every possible transition with count = 1
transition_counts = [
    [1] * num_tags
    for _ in range(num_tags)
]

# Add observed bigram counts
for (first, second), count in bigram_counts.items():

    if first not in tag_to_index or second not in tag_to_index:
        continue

    i = tag_to_index[first]
    j = tag_to_index[second]

    transition_counts[i][j] += count

# =========================================================
# Normalize each row
# =========================================================

transition_matrix = [
    [0.0] * num_tags
    for _ in range(num_tags)
]

for first in tag_inventory:

    i = tag_to_index[first]

    # Number of observed bigrams beginning with this tag
    #
    # + num_tags accounts for the add-one smoothing
    denominator = first_tag_counts[first] + num_tags

    for j in range(num_tags):

        transition_matrix[i][j] = (
            transition_counts[i][j] / denominator
        )

# =========================================================
# Verify probabilities
# =========================================================

print("\n==============================================")
print("ROW SUM VERIFICATION")
print("==============================================")

for i, tag in enumerate(tag_inventory):

    row_sum = sum(transition_matrix[i])

    print(
        f"{tag}: row sum = {row_sum:.6f}"
    )

# =========================================================
# Convert probability matrix to DataFrame
# =========================================================

bigram_df = pd.DataFrame(
    transition_matrix,
    columns=tag_inventory,
    index=tag_inventory
)

# =========================================================
# Print probability matrix
# =========================================================

print("\n==============================================")
print("SMOOTHED BIGRAM PROBABILITIES")
print("==============================================")

print(bigram_df)

# =========================================================
# Save probability matrix
# =========================================================

bigram_df.to_csv(probability_output_file)

print(
    f"\nSaved bigram probabilities to: "
    f"{probability_output_file}"
)

# =========================================================
# Final summary
# =========================================================

print("\n==============================================")
print("SUMMARY")
print("==============================================")

print(f"Number of sequences: {len(sequences)}")
print(f"Number of tags: {num_tags}")
print(f"Number of unique observed bigrams: {len(bigram_counts)}")

print(f"\nUnsmoothed counts:")
print(f"  {count_output_file}")

print(f"\nSmoothed probabilities:")
print(f"  {probability_output_file}")
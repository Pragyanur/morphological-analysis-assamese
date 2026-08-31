import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Load CSV
# =========================================================

csv_path = "results/error_analysis_n21_a21_epoch50_exponential_branch.csv"

df = pd.read_csv(csv_path)

total_sentences = len(df)


# =========================================================
# Count sentences for each number of incorrect words
# =========================================================

error_distribution = (
    df["incorrect_words"]
    .value_counts()
    .sort_index()
)


# =========================================================
# Calculate percentages
# =========================================================

percentages = (
    error_distribution / total_sentences * 100
)


# =========================================================
# Plot
# =========================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    error_distribution.index,
    error_distribution.values,
    color="gray"
)


# =========================================================
# Add count and percentage above each bar
# =========================================================

for bar, count, percentage in zip(
    bars,
    error_distribution.values,
    percentages.values
):

    plt.annotate(
        f"{count} ({percentage:.1f}%)",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height()
        ),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )


# =========================================================
# Labels
# =========================================================

plt.xlabel("Number of incorrect words")
plt.ylabel("Number of sentences")

plt.title(
    "Distribution of sentence-level ranking errors"
)

plt.xticks(
    error_distribution.index
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# =========================================================
# Save
# =========================================================

plt.savefig(
    "incorrect_words_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Load CSV
# =========================================================

csv_path = "results/error_analysis_n21_a21_epoch50_exponential_branch.csv"

df = pd.read_csv(csv_path)

# Sentence-level prediction is correct if there are no
# incorrect words in the predicted sequence.
df["correct"] = df["incorrect_words"] == 0


# =========================================================
# Helper function
# =========================================================

def calculate_accuracy(df, column, bins, labels):

    df = df.copy()

    df["bin"] = pd.cut(
        df[column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    result = (
        df.groupby("bin", observed=False)
          .agg(
              total=("correct", "size"),
              correct=("correct", "sum")
          )
    )

    result["accuracy"] = (
        result["correct"] / result["total"] * 100
    )

    return result.dropna()


# =========================================================
# 1. Sentence length
# =========================================================

length_bins = [
    0, 5, 10, 15, 20, 25, 30, 40, np.inf
]

length_labels = [
    "1–5",
    "6–10",
    "11–15",
    "16–20",
    "21–25",
    "26–30",
    "31–40",
    "41+"
]

length_result = calculate_accuracy(
    df,
    "length",
    length_bins,
    length_labels
)


# =========================================================
# 2. Number of ambiguous words
# =========================================================

ambiguous_bins = [
    0, 2, 4, 6, 8, 10, 15, 20, np.inf
]

ambiguous_labels = [
    "1–2",
    "3–4",
    "5–6",
    "7–8",
    "9–10",
    "11–15",
    "16–20",
    "21+"
]

ambiguous_result = calculate_accuracy(
    df,
    "ambiguous_words",
    ambiguous_bins,
    ambiguous_labels
)


# =========================================================
# 3. Number of candidate analyses
# =========================================================

candidate_bins = [
    0,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    np.inf
]

candidate_labels = [
    "1–10",
    "11–100",
    "101–1K",
    "1K–10K",
    "10K–100K",
    "100K–1M",
    "1M–10M",
    "10M–100M",
    "100M–1B",
    ">1B"
]

candidate_result = calculate_accuracy(
    df,
    "candidate_analyses",
    candidate_bins,
    candidate_labels
)


# =========================================================
# Print results
# =========================================================

print("\n==============================")
print("SENTENCE LENGTH")
print("==============================")
print(length_result.round(2))

print("\n==============================")
print("AMBIGUOUS WORDS")
print("==============================")
print(ambiguous_result.round(2))

print("\n==============================")
print("CANDIDATE ANALYSES")
print("==============================")
print(candidate_result.round(2))


# =========================================================
# Plot all three analyses
# =========================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 3.5)
)


# ---------------------------------------------------------
# Plot 1: Sentence length
# ---------------------------------------------------------

x = np.arange(len(length_result))

axes[0].plot(
    x,
    length_result["accuracy"],
    marker="o",
    linewidth=2
)

# Add number of sentences above each point
for i, (accuracy, total) in enumerate(
    zip(length_result["accuracy"], length_result["total"])
):
    axes[0].annotate(
        f"{total}",
        (i, accuracy),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

axes[0].set_xticks(x)
axes[0].set_xticklabels(
    length_result.index,
    rotation=45,
    ha="right"
)

axes[0].set_xlabel("Sentence length")
axes[0].set_ylabel("Ranking accuracy (%)")
axes[0].set_title("(a) Sentence length")

axes[0].set_ylim(0, 105)
axes[0].grid(
    axis="y",
    alpha=0.3
)


# ---------------------------------------------------------
# Plot 2: Ambiguous words
# ---------------------------------------------------------

x = np.arange(len(ambiguous_result))

axes[1].plot(
    x,
    ambiguous_result["accuracy"],
    marker="o",
    linewidth=2
)

# Add number of sentences above each point
for i, (accuracy, total) in enumerate(
    zip(ambiguous_result["accuracy"], ambiguous_result["total"])
):
    axes[1].annotate(
        f"{total}",
        (i, accuracy),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    ambiguous_result.index,
    rotation=45,
    ha="right"
)

axes[1].set_xlabel("Number of ambiguous words")
axes[1].set_ylabel("Ranking accuracy (%)")
axes[1].set_title("(b) Ambiguous words")

axes[1].set_ylim(0, 105)
axes[1].grid(
    axis="y",
    alpha=0.3
)


# ---------------------------------------------------------
# Plot 3: Candidate analyses
# ---------------------------------------------------------

x = np.arange(len(candidate_result))

axes[2].plot(
    x,
    candidate_result["accuracy"],
    marker="o",
    linewidth=2
)

# Add number of sentences above each point
for i, (accuracy, total) in enumerate(
    zip(candidate_result["accuracy"], candidate_result["total"])
):
    axes[2].annotate(
        f"{total}",
        (i, accuracy),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

axes[2].set_xticks(x)
axes[2].set_xticklabels(
    candidate_result.index,
    rotation=45,
    ha="right"
)

axes[2].set_xlabel("Number of candidate sequences")
axes[2].set_ylabel("Ranking accuracy (%)")
axes[2].set_title("(c) Candidate sequences")

axes[2].set_ylim(0, 105)
axes[2].grid(
    axis="y",
    alpha=0.3
)


# =========================================================
# Final layout
# =========================================================

plt.tight_layout()

plt.savefig(
    "sentence_level_error_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
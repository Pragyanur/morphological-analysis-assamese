import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import MorphoTransformer

def validate_ranking_weighted(model, val_groups, device):

    model.eval()

    total_weighted_score = 0
    total_possible_weight = 0

    total_correct_words = 0
    total_words = 0

    with torch.no_grad():

        for group in val_groups:
            candidates = [group['gold']] + group['negatives']
            num_candidates = len(candidates)
            weight = num_candidates
            batch_x = torch.stack(candidates).to(device)
            mask = torch.all(batch_x == 0, dim=-1)
            logits = model(batch_x, mask=mask).squeeze(1).cpu().numpy()
            best_idx = np.argmax(logits)

            # Sentence-level evaluation
            if best_idx == 0:
                total_weighted_score += weight
            total_possible_weight += weight

            # Word-level evaluation
            gold_mask = ~mask[0]
            word_matches = torch.all(
                batch_x[best_idx] == batch_x[0],
                dim=-1
            )
            total_correct_words += (
                word_matches & gold_mask
            ).sum().item()
            total_words += gold_mask.sum().item()

    ranking_acc = (
        total_weighted_score / total_possible_weight
        if total_possible_weight > 0 else 0
    )

    word_acc = (
        total_correct_words / total_words
        if total_words > 0 else 0
    )

    return ranking_acc, word_acc

import pandas as pd
import numpy as np
import torch


def validate_ranking_ambiguous_words(model, val_groups, device):
     """
     Validates by checking:
     1. Sentence-level Ranking Acc.
     2. Word-level Accuracy ONLY for ambiguous words (words where candidates disagree).
     """
     model.eval()
     correct_rankings = 0

     with torch.no_grad():
          for group in val_groups:
            candidates = [group['gold']] + group['negatives']
            batch_x = torch.stack(candidates).to(device) # Shape: (Num_Candidates, 30, 64)
            
            # Masking: True for padding (where all features are 0)
            mask = torch.all(batch_x == 0, dim=-1)
            
            # Get Scores
            scores = model(batch_x, mask=mask).squeeze(1).cpu().numpy()
            best_idx = np.argmax(scores)
            
            # 1. Sentence-Level Evaluation (Index 0 is always Gold)
            if best_idx == 0:
                 correct_rankings += 1
            
     ranking_acc = correct_rankings / len(val_groups)
     return ranking_acc

MODEL_DIM = 32
HEADS = 4
LAYERS = 3
LINEAR = 32
DIM_FF = 64
WINDOW_SIZE = 40
BATCH_SIZE = 128
AUG_FACTOR = 3
NEG_RATIO = 3

def run_training(
        X_train,
        y_train,
        val_groups,
        epochs=100,
        w_size=WINDOW_SIZE,
        heads=HEADS,
        layers=LAYERS,
        linear_dim=LINEAR,
        feedforward_dim=DIM_FF,
        model_dim=MODEL_DIM,
        batch_size=BATCH_SIZE,
        Na=AUG_FACTOR,
        Nn=NEG_RATIO
    ):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Training Dataloader
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    mask_train = torch.all(X_train_t == 0, dim=-1)
    train_ds = TensorDataset(X_train_t, y_train_t, mask_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 2. Setup Model
    model = MorphoTransformer(
        window_size=w_size, 
        nhead=heads, 
        num_layers=layers, 
        linear=linear_dim, 
        feedforward_dim=feedforward_dim,
        model_dim=model_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss_history = []
    ranking_history = []
    
    # 3. Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx, by, bm in train_loader:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            
            optimizer.zero_grad()
            logits = model(bx, mask=bm)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 4. Perform Ranking and Word Accuracy Validation
        ranking_acc = validate_ranking_ambiguous_words(
        # ranking_acc, word_acc, pipe_acc = validate_ranking_ambiguous_words(
            model=model,
            val_groups=val_groups,
            device=device
            )

        loss_history.append(epoch_loss/len(train_loader))
        ranking_history.append(ranking_acc)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Ranking Acc: {ranking_acc:.2%} | "
        )
    
    plot_unified_training_history(
        loss_history,
        ranking_history,
        # word_history,
        # pipe_history,
        Nn=Nn,
        Na=Na,
        model_dim=model_dim
    )
    
    return model


import matplotlib.pyplot as plt


def plot_unified_training_history(losses, ranking_accs, Nn, Na, model_dim=32):
    """Plots all training histories in a single clear, dual-scaled visualization window

    and saves it to disk as an image file.
    Left Axis: Neural Optimization Loss (Red)
    Right Axis: Operational Accuracy Tracks (Blue, Purple, Orange)
    """
    epochs = range(1, len(losses) + 1)

    # Initialize a single plot figure
    fig, ax_loss = plt.subplots(figsize=(10, 6))

    # =========================================================
    # 1. Left Y-Axis: Loss Evolution (Optimization Scale)
    # =========================================================
    loss_color = "#d62728"  # Muted professional red
    line_loss = ax_loss.plot(
        epochs,
        losses,
        color=loss_color,
        linestyle="--",
        linewidth=2,
        label="Training Loss",
    )

    ax_loss.set_xlabel("Epochs", fontsize=11, fontweight="bold")
    ax_loss.set_ylabel("Loss", color=loss_color, fontsize=11, fontweight="bold")
    ax_loss.tick_params(axis="y", labelcolor=loss_color)

    # Add defensive padding calculation to keep loss curves off extreme boundaries
    loss_min, loss_max = min(losses), max(losses)
    loss_margin = max((loss_max - loss_min) * 0.05, 0.005)
    ax_loss.set_ylim(loss_min - loss_margin, loss_max + loss_margin)

    # =========================================================
    # 2. Right Y-Axis: Accuracies (Unified Pipeline Scale)
    # =========================================================
    ax_acc = ax_loss.twinx()

    # Track 1: Ranking Accuracy (Blue)
    line_rank = ax_acc.plot(
        epochs, ranking_accs, color="#1f77b4", linewidth=2.5, label="Ranking Acc."
    )

    ax_acc.set_ylabel(
        "Evaluation Accuracy Profile",
        color="#2c3e50",
        fontsize=11,
        fontweight="bold",
    )
    ax_acc.tick_params(axis="y", labelcolor="#2c3e50")

    # Extract structural boundaries across all 3 accuracies for the Y-limit
    # all_accs = ranking_accs + word_accs + pipe_accs
    all_accs = ranking_accs
    acc_min, acc_max = min(all_accs), max(all_accs)
    acc_margin = max((acc_max - acc_min) * 0.05, 0.03)
    ax_acc.set_ylim(
        max(0.0, acc_min - acc_margin), min(1.05, acc_max + acc_margin)
    )

    # Set y-axis ticks to print cleanly as percentages if data is stored as decimals (0.0-1.0)
    vals = ax_acc.get_yticks()
    ax_acc.set_yticklabels(["{:,.0%}".format(x) for x in vals])

    # =========================================================
    # 3. Structural Polish & Legends
    # =========================================================
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # Group every line pointer array smoothly into a centralized legend block
    all_lines = line_loss + line_rank
    all_labels = [line.get_label() for line in all_lines]

    # Places a horizontal legend cleanly above the title boundary
    ax_loss.legend(
        all_lines,
        all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=4,  # Spreads the 4 labels horizontally in a single row
        frameon=True,
    )

    plt.tight_layout()

    # =========================================================
    # 4. Save and Close File
    # =========================================================
    output_name = f"models/training_{Nn}_{Na}.png"
    # bbox_inches='tight' forces matplotlib to include the floating top legend safely
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close()  # Clear memory allocations after saving
    print(f"Graph safely written out to local file: {output_name}")
import pandas as pd
import torch
from create_training_validation import create_complete_morphology_dataset, create_validation_groups
from training import run_training
from morphology import vectorize_morphology, sentence_word_options
from sklearn.model_selection import train_test_split

def train_test_split_by_sentence(df, test_size=0.1, random_state=10):
    """
    Splits the dataframe into train and validation sets based on unique sentence IDs.
    This prevents data leakage between augmented/jittered versions of the same sentence.
    """
    # 1. Get unique sentence IDs
    unique_ids = df['sentence-id'].unique()
    
    # 2. Split the IDs themselves
    train_ids, val_ids = train_test_split(
        unique_ids, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 3. Filter the original dataframe using the split IDs
    train_df = df[df['sentence-id'].isin(train_ids)].copy()
    val_df = df[df['sentence-id'].isin(val_ids)].copy()
    
    print(f"Total Sentences: {len(unique_ids)}")
    print(f"Training Sentences: {len(train_ids)} ({len(train_df)} rows) (~{len(train_ids) * NEG_RATIO * AUG_FACTOR} samples)")
    print(f"Validation Sentences: {len(val_ids)} ({len(val_df)} rows)")
    
    return train_df, val_df

file_ext = ".csv"

DATASET_PATH = "dataset/1072-annotated-dataset-updated.csv"

EPOCHS = 5
NEG_RATIO = 2
AUG_FACTOR = 2
MODEL_DIM = 32
HEADS = 8
LAYERS = 1
LINEAR = 32
DIM_FF = 64
WINDOW_SIZE = 64
BATCH_SIZE = 1024
FEATURE_DIM = 64

RANDOM_STATE = 0
TEST_SIZE = 0.2

# for nn in [1, 2, 4, 16]:
#     for na in [1, 2, 4, 16]:


# 1. Split your data by sentence ID
train_df, val_df = train_test_split_by_sentence(pd.read_csv(DATASET_PATH), TEST_SIZE, RANDOM_STATE)

# 2. Create the jittered training tensors (Flattened for BCE loss)
X_train, y_train, _ = create_complete_morphology_dataset(train_df, vectorize_morphology, neg_ratio=NEG_RATIO, augment_factor=AUG_FACTOR)

# 3. Create the structured validation groups (Grouped for Ranking)
val_groups = create_validation_groups(val_df, vectorize_morphology)

# 4. Train the model
# In every epoch, the model trains on X_train (binary classification)
# Then validates on val_groups (ranking accuracy)
model = run_training(X_train, y_train, val_groups, epochs=EPOCHS, w_size=WINDOW_SIZE, heads=HEADS, layers=LAYERS, linear_dim=LINEAR, feedforward_dim=DIM_FF, batch_size=BATCH_SIZE, Na=AUG_FACTOR, Nn=NEG_RATIO)

# To save the weights only (Recommended)
torch.save(model.state_dict(), f"models/n{NEG_RATIO}_a{AUG_FACTOR}_epoch{EPOCHS}.pth")

params = {
    "MODEL_DIM" : MODEL_DIM,
    "HEADS" : HEADS,
    "LAYERS" : LAYERS,
    "LINEAR" : LINEAR,
    "DIM_FF" : DIM_FF,
    "WINDOW_SIZE" : WINDOW_SIZE,
    "BATCH_SIZE" : BATCH_SIZE,
    "FEATURE_DIM" : FEATURE_DIM
}

import json
from error_analysis import validation_analysis, export_single_analysis_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"models/n{NEG_RATIO}_a{AUG_FACTOR}_epoch{EPOCHS}_MODEL_PARAMS.json", "w") as f:
    json.dump(params, f, indent=" ")


sentences_data = validation_analysis(val_groups, model, device, sentence_word_options, vectorize_morphology, WINDOW_SIZE, FEATURE_DIM, candidate_stats_file=f"results/error_analysis_n{NEG_RATIO}_a{AUG_FACTOR}_epoch{EPOCHS}_exponential_branch.csv")
export_single_analysis_report(sentences_data, f"results/error_analysis_n{NEG_RATIO}_a{AUG_FACTOR}_epoch{EPOCHS}_exponential_branch.html")

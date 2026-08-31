import tkinter as tk
from tkinter import ttk
import pandas as pd
import ast
import os

WARP_LEN = 1260

INPUT_PATH = "resources/samples.csv"
OUTPUT_PATH = "resources/samples_checkpoint.csv"
df = pd.read_csv(INPUT_PATH)

root = tk.Tk()
root.title("Annotation Tool")
root.geometry("1260x720")

# Default values
current_index = 0
annotations = []
previous_choices = {}


# Load checkpoint if exists
if os.path.exists(OUTPUT_PATH):
    # print("file-found")
    checkpoint_df = pd.read_csv(OUTPUT_PATH)

    annotations = checkpoint_df.to_dict("records")

    for ann in annotations:
        previous_choices[ann["word"]] = ann["selected-index"]
        
    if len(checkpoint_df) > 0:
        last_sentence_id = checkpoint_df.iloc[-1]["sentence-id"]
        last_token_id = checkpoint_df.iloc[-1]["word-id"]
        print(last_sentence_id)
        # Find next row in original dataframe
        matches = df[
            (df["sentence-id"] == last_sentence_id) &
            (df["word-id"] == last_token_id)
        ]
        
        if len(matches) > 0:
            current_index = matches.index[0] + 1

print("Starting from index:", current_index)

sentence_label = tk.Label(
    root,
    text="",
    font=("Arial", 16),
    wraplength=WARP_LEN,
    justify="left"
)
sentence_label.pack(pady=20)

token_label = tk.Label(
    root,
    text="",
    font=("Arial", 20, "bold")
)
token_label.pack(pady=10)

selected_option = tk.IntVar(value=-1)

radio_buttons = []

# options_frame = tk.Frame(root)
# options_frame.pack(pady=20)

# =========================================================
# Scrollable options area
# =========================================================

container = tk.Frame(root)
container.pack(fill="both", expand=True, pady=10)

canvas = tk.Canvas(container)

scrollbar = ttk.Scrollbar(
    container,
    orient="vertical",
    command=canvas.yview
)

scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window(
    (0, 0),
    window=scrollable_frame,
    anchor="nw"
)

canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# This becomes your new options frame
options_frame = scrollable_frame

def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

status_label = tk.Label(root, text="")
status_label.pack()


def load_item():
    global current_index

    if current_index >= len(df):
        save_annotations()
        sentence_label.config(text="Annotation Completed!")
        token_label.config(text="")
        for rb in radio_buttons:
            rb.destroy()
        next_button.config(state=tk.DISABLED)
        return

    row = df.iloc[current_index]

    sentence_label.config(
        text=f"Sentence:\n{row['sentence']}"
    )

    token_label.config(
        text=f"Token: {row['word']}"
    )

    # selected_option.set(-1)
    token = row["word"]

    # Convert string representation to list
    options = ast.literal_eval(row["options"])

    if token in previous_choices:
        selected_option.set(previous_choices[token])

    # If there are exactly 2 options,
    # automatically select the first option
    elif len(options) == 2:
        selected_option.set(0)

    else:
        selected_option.set(-1)
        
    # Remove old buttons
    for rb in radio_buttons:
        rb.destroy()

    radio_buttons.clear()

    # Convert string representation to list
    options = ast.literal_eval(row["options"])

    # Dynamically create buttons
    for idx, option in enumerate(options):

        rb = tk.Radiobutton(
            options_frame,
            text=f"{idx} : {option}",
            variable=selected_option,
            value=idx,
            font=("Arial", 14),
            wraplength=WARP_LEN,   # adjust as needed
            justify="left",
            anchor="w"
        )

        rb.pack(anchor="w", fill="x", pady=2)
        
        radio_buttons.append(rb)

    status_label.config(
        text=f"{current_index + 1} / {len(df)}"
    )


def next_item():
    global current_index

    choice = selected_option.get()

    if choice == -1:
        return

    row = df.iloc[current_index]

    annotations.append({
        "sentence-id": row["sentence-id"],
        "word-id": row["word-id"],
        "sentence": row["sentence"],
        "word": row["word"],
        "options": row["options"],
        "selected-index": choice
    })
    
    previous_choices[row["word"]] = choice  
    
    # SAVE AFTER EVERY ANNOTATION
    save_annotations()

    current_index += 1
    load_item()


# def save_annotations():
#     out_df = pd.DataFrame(annotations)
#     # out_df.to_csv(f"annotation/annotations_checkpoint_opts_{MAX_OPTS}.csv", index=False)
#     out_df.to_csv(checkpoint_file)

def save_annotations():
    out_df = pd.DataFrame(annotations)
    out_df.to_csv(OUTPUT_PATH, index=False)


next_button = ttk.Button(
    root,
    text="Next",
    command=next_item
)
next_button.pack(pady=20)

load_item()

root.mainloop()
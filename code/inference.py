import json
import torch
import numpy as np
import itertools
from model import MorphoTransformer
from morphology import sentence_word_options, vectorize_morphology

def run_inference(sentence, morphology_generator, vectorize_func, model_path, window_size, feature_dim, heads, layers, model_dim, linear, dim_ff):
    """
    sentence: String (Assamese sentence)
    morphology_generator: Function that returns a list of lists. 
                         Example: [ [word1_opt1, word1_opt2], [word2_opt1], ... ]
    vectorize_func: Your bit-mapping function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = MorphoTransformer(feature_dim, window_size, heads, layers, model_dim, linear, dim_ff).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Generate Options for each word
    # word_options is a list of lists of vectors
    sentence_options = morphology_generator(sentence)
    
    # Vectorize and clean shapes
    vectorized_options = []
    for word_options in sentence_options: # word_options is [opt1, opt2...]
        word_choices = []
        for opt in word_options:
            v = vectorize_func(opt)
            # v = get_clean_vec(v) # Optional: Use the shape-guard logic from earlier
            
            # Store as (vector, raw_label)
            word_choices.append((v, opt))
        vectorized_options.append(word_choices)

    # 3. Create all possible sequence combinations
    all_possible_combinations = list(itertools.product(*vectorized_options))
    
    # 4. Prepare Batch for Model
    padding_vec = np.zeros(feature_dim, dtype=np.float32)
    batch_list = []
    all_labels = [] # To store the sequences of tags
    
    for combo in all_possible_combinations:
        # combo is ((v1, tag1), (v2, tag2), ...)
        seq_vectors = [item[0] for item in combo]
        seq_labels = [item[1] for item in combo]
        
        seq_len = len(seq_vectors)
        if seq_len > window_size:
            seq_vectors = seq_vectors[:window_size]
            seq_len = window_size
        
        padded = list(seq_vectors) + ([padding_vec] * (window_size - seq_len))
        batch_list.append(np.stack(padded))
        all_labels.append(seq_labels)

    # Convert to Tensor
    input_tensor = torch.FloatTensor(np.array(batch_list)).to(device)
    mask = torch.all(input_tensor == 0, dim=-1)

    # 5. Get Scores
    with torch.no_grad():
        # Processing in chunks if the number of combinations is massive
        logits = model(input_tensor, mask=mask).squeeze(1)
        scores = torch.sigmoid(logits).cpu().numpy()

    # 6. Find Winner
    best_idx = np.argmax(scores)
    
    return {
        "best_score": float(scores[best_idx]),
        "total_combinations": len(all_possible_combinations),
        "best_sequence_index": int(best_idx),
        "best_sequence_tags": all_labels[best_idx] # This is your non-vectorized winner
    }

MODEL_PATH = "models/n21_a21_epoch50.pth"

with open("models/n21_a21_epoch50_MODEL_PARAMS.json", "r") as f:
    params = json.load(f)

text = "শীঘ্ৰেই ৰেগিং কাণ্ডৰ লগত জড়িত কেবাগৰাকী জ্যেষ্ঠ ছাত্ৰ গ্ৰেপ্তাৰ হোৱাৰ সম্ভাৱনা আছে ।"

result = run_inference(text, sentence_word_options, vectorize_morphology, MODEL_PATH, params["WINDOW_SIZE"], params["FEATURE_DIM"], params["HEADS"], params["LAYERS"], params["MODEL_DIM"], params["LINEAR"], params["DIM_FF"])
print(result)

# {'best_score': 0.26195940375328064, 'total_combinations': 216, 'best_sequence_index': 55, 'best_sequence_tags': [[['শীঘ্ৰে', 'Adverb'], ['ই', 'emph']], [['ৰেগিং', 'Verbal Noun']], [['কাণ্ড', 'Abstract Noun'], ['ৰ', 'gen']], [['লগত', 'Conjuction']], [['জড়িত', 'Proper Adj.']], [['কেব', 'Verbable'], ['া', 'adj_comp'], ['গৰাকী', 'number']], [['জ্যেষ্ঠ', 'Proper Adj.']], [['ছাত্ৰ', 'Common Noun']], [['গ্ৰেপ্তাৰ', 'Verbal Noun']], [['হ', 'Verb-Intran.'], ['োৱা', 'adj_comp'], ['ৰ', 'gen']], [['সম্ভাৱনা', 'Abstract Noun']], [['আছ', 'Verb-Intran.'], ['ে', 'present-3a']], [['।', 'symbol']]]}
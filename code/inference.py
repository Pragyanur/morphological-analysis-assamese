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

file_ext_model = "_MODEL.pth"

MODEL_PATH = f"annotation/annotated-dataset/combined_misalignment-free{file_ext_model}"
# MODEL_PATH = "models/sent-word-opts-label-4-excl-chatgpt.csv_MODEL.pth"

with open(f"{MODEL_PATH[:len(MODEL_PATH) - len(file_ext_model)]}_MODEL_PARAMS.json", "r") as f:
    params = json.load(f)

# Example Usage:
# result = run_inference("মোৰ শিক্ষা জীৱনৰ আৰম্ভণি হয় মাৰ্ঘেৰিটাত ।", sentence_word_options, vectorize_morphology, MODEL_PATH)
result = run_inference("মোৰ শিক্ষা জীৱনৰ আৰম্ভণি জাকজমকতাৰে হয় মাৰ্ঘেৰিটাত ।", sentence_word_options, vectorize_morphology, MODEL_PATH, params["WINDOW_SIZE"], params["FEATURE_DIM"], params["HEADS"], params["LAYERS"], params["MODEL_DIM"], params["LINEAR"], params["DIM_FF"])
print(result)

# annotated => {'best_score': 0.5757861733436584, 'total_combinations': 540, 'best_sequence_index': 427, 'best_sequence_tags': 
# [[['ডাঙৰ', 'Proper Adj.'], ['ে', 'present-3a']], [['আৰু', 'Conjuction']], [['লগৰীয়া', 'Common Noun'], ['বিলাক', 'number'], ['ে', 'emph']], [['ভালপোৱা', 'Abstract Noun'], ['টো', 'number']], [['শিশু', 'Proper Noun'], ['ৱে', 'nom']], [['বিচাৰ', 'Verb-Trans.'], ['ে', 'present-3a']], [['।', 'symbol']]]}
# chatgpt =>  {'best_score': 0.485487699508667, 'total_combinations': 540, 'best_sequence_index': 228, 'best_sequence_tags': 
# [[['ডাঙৰ', 'Proper Adj.'], ['ে', 'nom']], [['আৰু', 'Adverb']], [['লগৰীয়া', 'Common Noun'], ['বিলাক', 'number'], ['ে', 'nom']], [['ভা', 'Abstract Noun'], ['ল', 'derivational'], ['', 'Material Noun'], ['পোৱা', 'Proper Adj.'], ['টো', 'number']], [['শিশু', 'Common Noun'], ['ৱে', 'nom']], [['বিচাৰ', 'Verb-Trans.'], ['ে', 'when_comp']], [['।', 'symbol']]]}
# {'best_score': 0.3674145042896271, 'total_combinations': 180, 'best_sequence_index': 95, 'best_sequence_tags': 
# [[['মোৰ', 'Pronoun'], ['', 'gen']], [['শিক্ষা', 'Abstract Noun']], [['জীৱন', 'Abstract Noun'], ['ৰ', 'gen']], [['আৰম্ভণি', 'Abstract Noun']], [['হ', 'Verb-Intran.'], ['য়', 'present-3a']], [['মাৰ্ঘেৰিটা', 'Proper Noun'], ['ত', 'loc']], [['।', 'symbol']]]}
# chatgpt => {'best_score': 0.6373212933540344, 'total_combinations': 180, 'best_sequence_index': 55, 'best_sequence_tags': 
# [[['মোৰ', 'Abstract Noun']], [['শিক্ষা', 'Abstract Noun']], [['জীৱ', 'Common Noun'], ['ন', 'Proper Adj.'], ['ৰ', 'gen']], [['আৰম্ভণি', 'Abstract Noun']], [['হয়', 'Common Noun']], [['মাৰ্ঘেৰিটা', 'Proper Noun'], ['ত', 'loc']], [['।', 'symbol']]]}


# {'best_score': 0.03235650435090065, 'total_combinations': 540, 'best_sequence_index': 347, 'best_sequence_tags': 
# [[['মোৰ', 'Pronoun'], ['', 'gen']], [['শিক্ষা', 'Abstract Noun']], [['জীৱ', 'Abstract Noun'], ['ন', 'Abstract Noun'], ['ৰ', 'gen']], [['জাকজমক', 'Adjective Adj.'], ['তা', 'derivational'], ['', 'Abstract Noun'], ['ৰে', 'inst']], [['আৰম্ভণি', 'Abstract Noun']], [['হ', 'Verb-Intran.'], ['য়', 'present-3a']], [['মাৰ্ঘেৰিটা', 'Proper Noun'], ['ত', 'loc']], [['।', 'symbol']]]}
# {'best_score': 0.03834787756204605, 'total_combinations': 540, 'best_sequence_index': 335, 'best_sequence_tags': 
# [[['মোৰ', 'Pronoun'], ['', 'gen']], [['শিক্ষা', 'Abstract Noun']], [['জীৱ', 'Abstract Noun'], ['ন', 'Abstract Noun'], ['ৰ', 'gen']], [['আৰম্ভণি', 'Abstract Noun']], [['জাকজমক', 'Adjective Adj.'], ['তা', 'derivational'], ['', 'Abstract Noun'], ['ৰে', 'inst']], [['হ', 'Verb-Intran.'], ['য়', 'present-3a']], [['মাৰ্ঘেৰিটা', 'Proper Noun'], ['ত', 'loc']], [['।', 'symbol']]]}
# {'best_score': 0.30334147810935974, 'total_combinations': 540, 'best_sequence_index': 281, 'best_sequence_tags': 
# [[['মোৰ', 'Pronoun'], ['', 'gen']], [['শিক্ষা', 'Abstract Noun']], [['জীৱন', 'Abstract Noun'], ['ৰ', 'gen']], [['আৰম্ভণি', 'Abstract Noun']], [['জাকজমক', 'Adjective Adj.'], ['তা', 'derivational'], ['', 'Abstract Noun'], ['ৰে', 'inst']], [['হ', 'Verb-Intran.'], ['য়', 'present-3a']], [['মাৰ্ঘেৰিটা', 'Proper Noun'], ['ত', 'loc']], [['।', 'symbol']]]}

# with derivation as weight, augment 3 and negative 3
# {'best_score': 0.1567164957523346, 'total_combinations': 540, 'best_sequence_index': 281, 'best_sequence_tags': 
# [[['মোৰ', 'Pronoun'], ['', 'gen']], [['শিক্ষা', 'Abstract Noun']], [['জীৱন', 'Abstract Noun'], ['ৰ', 'gen']], [['আৰম্ভণি', 'Abstract Noun']], [['জাকজমক', 'Adjective Adj.'], ['তা', 'derivational'], ['', 'Abstract Noun'], ['ৰে', 'inst']], [['হ', 'Verb-Intran.'], ['য়', 'present-3a']], [['মাৰ্ঘেৰিটা', 'Proper Noun'], ['ত', 'loc']], [['।', 'symbol']]]}
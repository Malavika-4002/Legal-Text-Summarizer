#src/sbd/inference.py

import joblib
import re
import torch
import pandas as pd
import os
import json

# --- CHANGED: Imports ---
from src.sbd.model_cnn import LegalSBD_CNN
from src.sbd.feature_extractor import token_to_features, add_neighboring_token_features

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'sbd_weights')
CNN_PATH = os.path.join(MODEL_DIR, 'cnn_model.pth')
CRF_PATH = os.path.join(MODEL_DIR, 'crf_hybrid_model.joblib')
VOCAB_PATH = os.path.join(MODEL_DIR, 'char_vocab.json')

# --- GLOBAL VARIABLES ---
CONTEXT_WINDOW_SIZE = 6
DELIMITERS = {'.', '?', '!', ';', ':'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOAD MODELS ONCE (When file is imported) ---
print("Loading SBD Models...")
try:
    # 1. Load Vocab
    with open(VOCAB_PATH, 'r') as f:
        char_to_idx = json.load(f)
    
    # 2. Load CNN
    cnn_model = LegalSBD_CNN(len(char_to_idx), 128, 6, 5, 250, 0.2).to(device)
    cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
    cnn_model.eval()

    # 3. Load CRF
    crf_model = joblib.load(CRF_PATH)
    print("✅ SBD Models Loaded.")
except Exception as e:
    print(f"⚠️ Warning: Could not load SBD models. Error: {e}")
    print("You might need to run src/sbd/train.py first.")
    cnn_model, crf_model = None, None


# --- YOUR EXACT FUNCTIONS ---
def get_cnn_prediction_from_context(text, token_start_idx):
    token = text[token_start_idx]
    if token not in DELIMITERS: return 0.0
    start_left = max(0, token_start_idx - CONTEXT_WINDOW_SIZE)
    sample_text = text[start_left : token_start_idx] + token + text[token_start_idx + 1 : token_start_idx + 1 + CONTEXT_WINDOW_SIZE]
    max_len = (CONTEXT_WINDOW_SIZE * 2) + 1
    pad_idx = char_to_idx['<PAD>']
    indexed_text = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in sample_text]
    padded_text = indexed_text[:max_len] + [pad_idx] * (max_len - len(indexed_text))
    text_tensor = torch.tensor([padded_text], dtype=torch.long).to(device)
    with torch.no_grad(): return cnn_model(text_tensor).item()

def get_sentences(text):
    """
    Sentence segmentation using CNN + CRF with legal-text safeguards.
    """
    if not crf_model:
        return [text]

    tokens_with_spans = [
        (m.group(0), m.start(), m.end())
        for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)
    ]

    if not tokens_with_spans:
        return []

    sentence_features = []
    for token, start, end in tokens_with_spans:
        features = token_to_features(token, text, start, end)

        if token in DELIMITERS:
            cnn_prob = get_cnn_prediction_from_context(text, start)
            features['cnn_prob'] = round(cnn_prob, 4)

        sentence_features.append(features)

    sentence_features = add_neighboring_token_features(sentence_features)
    labels = crf_model.predict([sentence_features])[0]

    sentences = []
    current_idx = 0

    for i, label in enumerate(labels):
        if label == 'B':

            # ✅ LEGAL-TEXT SAFETY GATE
            if i + 1 < len(tokens_with_spans):
                next_token = tokens_with_spans[i + 1][0]
                if next_token and next_token[0].islower():
                    continue  # suppress false split

            end_idx = tokens_with_spans[i][2]
            start_idx = tokens_with_spans[current_idx][1]
            sentences.append(text[start_idx:end_idx].strip())
            current_idx = i + 1

    if current_idx < len(tokens_with_spans):
        sentences.append(
            text[tokens_with_spans[current_idx][1]:].strip()
        )

    return sentences

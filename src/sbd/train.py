import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from tqdm import tqdm
import re
import joblib 
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- CHANGED: Updated Imports for new folder structure ---
from src.sbd.model_cnn import LegalSBD_CNN, SBDDataset
from src.sbd.feature_extractor import token_to_features, add_neighboring_token_features

# --- CHANGED: Configuration with Absolute Paths ---
# This ensures Python finds files regardless of where you run the command
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'sbd_weights')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

CONTEXT_WINDOW_SIZE = 6
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_model.pth')
CRF_BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, 'crf_baseline_model.joblib')
CRF_HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, 'crf_hybrid_model.joblib')
PERFORMANCE_REPORT_PATH = os.path.join(MODEL_DIR, 'performance_report.json')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')

DELIMITERS = {'.', '?', '!', ';', ':'}
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 250
NUM_FILTERS = 6
KERNEL_SIZE = 5
DROPOUT_PROB = 0.2

# --- YOUR HELPER FUNCTIONS (Kept Exact) ---
def load_cnn_model(model_path, vocab_size, device):
    model = LegalSBD_CNN(vocab_size, EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZE, HIDDEN_DIM, DROPOUT_PROB).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_cnn_prediction_from_context(text, token_start_idx, cnn_model, char_to_idx, device):
    token = text[token_start_idx]
    if token not in DELIMITERS: return 0.0
    start_left = max(0, token_start_idx - CONTEXT_WINDOW_SIZE)
    sample_text = text[start_left : token_start_idx] + token + text[token_start_idx + 1 : token_start_idx + 1 + CONTEXT_WINDOW_SIZE]
    max_len = (CONTEXT_WINDOW_SIZE * 2) + 1
    pad_idx = char_to_idx['<PAD>']
    indexed_text = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in sample_text]
    padded_text = indexed_text[:max_len] + [pad_idx] * (max_len - len(indexed_text))
    text_tensor = torch.tensor([padded_text], dtype=torch.long).to(device)
    with torch.no_grad(): prediction = cnn_model(text_tensor).item()
    return prediction

def prepare_data_for_crf(file_path, cnn_model=None, char_to_idx=None, device=None):
    # (Your exact logic here)
    X, y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Preparing CRF from {os.path.basename(file_path)}"):
            data = json.loads(line)
            text = data['text']
            try: true_boundary_offsets = {span['end'] for span in data['spans']}
            except KeyError: continue
            tokens_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)]
            if not tokens_with_spans: continue
            
            sentence_features, labels = [], []
            for token, start, end in tokens_with_spans:
                features = token_to_features(token, text, start, end)
                if cnn_model and token in DELIMITERS:
                    delimiter_char_index = text.find(token, start)
                    if delimiter_char_index != -1:
                         cnn_prob = get_cnn_prediction_from_context(text, delimiter_char_index, cnn_model, char_to_idx, device)
                         features['cnn_prob'] = round(cnn_prob, 4)
                sentence_features.append(features)
                labels.append('B' if end in true_boundary_offsets and token in DELIMITERS else 'O')
            X.append(add_neighboring_token_features(sentence_features))
            y.append(labels)
    return X, y

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("--- Part 1: Retraining CNN ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if data exists
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ ERROR: Could not find train_data.csv at {TRAIN_DATA_PATH}")
        print("Please move your CSV file to 'data/processed/'")
        exit()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    all_chars = set().union(*train_df['left_context'].astype(str), *train_df['delimiter'].astype(str), *train_df['right_context'].astype(str))
    char_to_idx = {c: i+2 for i, c in enumerate(sorted(list(all_chars)))}
    char_to_idx.update({'<PAD>': 0, '<UNK>': 1})
    
    # Save vocab for Inference (Important for Phase 2!)
    with open(os.path.join(MODEL_DIR, 'char_vocab.json'), 'w') as f:
        json.dump(char_to_idx, f)

    dataset = SBDDataset(TRAIN_DATA_PATH, char_to_idx, (CONTEXT_WINDOW_SIZE * 2) + 1)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    cnn = LegalSBD_CNN(len(char_to_idx), EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZE, HIDDEN_DIM, DROPOUT_PROB).to(device)
    opt = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    crit = nn.BCELoss()
    
    cnn.train()
    for _ in range(2): # Reduced epochs for testing migration. Increase back to 15 later.
        for text, labels in tqdm(loader):
            opt.zero_grad()
            loss = crit(cnn(text.to(device)), labels.to(device))
            loss.backward()
            opt.step()
    torch.save(cnn.state_dict(), CNN_MODEL_PATH)
    print(f"CNN Saved to {CNN_MODEL_PATH}")

    print("\n--- Part 2: Training Hybrid CRF ---")
    # Note: Ensure these JSONL files are in data/raw/
    raw_train_files = [os.path.join(BASE_DIR, 'data/raw/CD_bva.jsonl')] 
    
    X_train, y_train = [], []
    for f in raw_train_files:
        if os.path.exists(f):
            x, y = prepare_data_for_crf(f, cnn_model=cnn, char_to_idx=char_to_idx, device=device)
            X_train.extend(x); y_train.extend(y)
    
    if X_train:
        crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=50, all_possible_transitions=True)
        crf.fit(X_train, y_train)
        joblib.dump(crf, CRF_HYBRID_MODEL_PATH)
        print(f"Hybrid CRF Saved to {CRF_HYBRID_MODEL_PATH}")
    else:
        print("Skipped CRF training (No JSONL data found in data/raw/)")
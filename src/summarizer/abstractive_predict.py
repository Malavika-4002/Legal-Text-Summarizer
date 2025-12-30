# src/summarizer/abstractive_predict.py

import torch
import torch.nn.functional as F
import sentencepiece as spm
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.summarizer.model_seq2seq import Encoder, DecoderWithAttention, BahdanauAttention, Seq2Seq

class LegalAbstractor:
    def __init__(self, model_path, spm_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_path)
        self.vocab_size = self.sp.get_piece_size()
        
        # Hyperparams must match training
        emb_dim, hidden_dim = 256, 256
        self.bos_id, self.eos_id, self.pad_id = 2, 3, 0
        
        encoder = Encoder(self.vocab_size, emb_dim, hidden_dim, pad_id=self.pad_id)
        attention = BahdanauAttention(hidden_dim)
        decoder = DecoderWithAttention(self.vocab_size, emb_dim, hidden_dim, attention, pad_id=self.pad_id)
        
        self.model = Seq2Seq(encoder, decoder, pad_id=self.pad_id, device=self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def summarize(self, text, beam_width=5, max_len=50, repetition_penalty=1.8, no_repeat_ngram_size=2):
        ids = [self.bos_id] + self.sp.encode(text) + [self.eos_id]
        src_tensor = torch.tensor([ids], dtype=torch.long).to(self.device)
        src_len = torch.tensor([len(ids)]).to(self.device)

        with torch.no_grad():
            encoder_outputs, h_n, c_n = self.model.encoder(src_tensor, src_len)
            beams = [(0.0, [self.bos_id], h_n, c_n)]
            
            for _ in range(max_len):
                candidates = []
                for score, current_ids, h, c in beams:
                    if current_ids[-1] == self.eos_id:
                        candidates.append((score, current_ids, h, c))
                        continue
                    
                    input_token = torch.tensor([current_ids[-1]], device=self.device)
                    logits, next_h, next_c, _ = self.model.decoder(input_token, h, c, encoder_outputs)
                    
                    # Apply Repetition Penalty
                    for prev_id in set(current_ids):
                        logits[0, prev_id] /= repetition_penalty

                    log_probs = torch.log_softmax(logits, dim=1)
                    
                    # N-Gram Blocking
                    if no_repeat_ngram_size > 0 and len(current_ids) >= no_repeat_ngram_size:
                        ngram = current_ids[-(no_repeat_ngram_size-1):]
                        for i in range(len(current_ids) - no_repeat_ngram_size + 1):
                            if current_ids[i:i + no_repeat_ngram_size - 1] == ngram:
                                forbidden_id = current_ids[i + no_repeat_ngram_size - 1]
                                log_probs[0, forbidden_id] = -1e10

                    top_v, top_i = log_probs.topk(beam_width)
                    for i in range(beam_width):
                        # Length penalty to encourage conclusion
                        lp = ((5 + len(current_ids) + 1) / 6) ** 0.6
                        next_score = (score + top_v[0, i].item()) / lp
                        candidates.append((next_score, current_ids + [top_i[0, i].item()], next_h, next_c))
                
                beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
                if all(b[1][-1] == self.eos_id for b in beams): break

        cleaned_ids = [idx for idx in beams[0][1] if idx not in [self.bos_id, self.eos_id, self.pad_id]]
        return self.sp.decode(cleaned_ids).strip().capitalize()
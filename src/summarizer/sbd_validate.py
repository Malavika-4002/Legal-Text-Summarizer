# src/summarizer/sbd_validate.py
import os
import json
import random
import re
import math
import argparse
from collections import Counter

import pandas as pd

# Use your SBD wrapper
from src.summarizer.sentence_splitter import split_into_sentences

# Output path
OUT_JSONL = "data/processed/sbd_diagnostics.jsonl"

# Heuristics
LONG_SENT_WORDS = 200   # If SBD returns sentences longer than this -> likely missed split
SAMPLE_SIZE = 50        # default sample size


def simple_regex_split(text):
    # basic fallback: split on sentence-ending punctuation followed by space+capital OR newline
    # keep the punctuation with the sentence
    parts = re.split(r'(?<=[\.\?\!;:])\s+(?=[A-Z0-9"\'])', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def find_false_positive_splits(original_text, sbd_sentences):
    """
    Find likely false-positive splits produced by SBD:
      - a split occurred at a delimiter but the following character (in original text) is lowercase
    We return context snippets showing the end of previous sentence and start of next sentence.
    """
    flags = []
    # Build a rolling search to locate each sentence in the original text
    pos = 0
    for i in range(len(sbd_sentences)-1):
        prev_sent = sbd_sentences[i]
        next_sent = sbd_sentences[i+1]
        # find prev_sent starting from pos
        try:
            start_prev = original_text.index(prev_sent, pos)
            end_prev = start_prev + len(prev_sent)
            # check next char in original text after end_prev (if exists and is not whitespace)
            # find first non-space char after end_prev
            j = end_prev
            while j < len(original_text) and original_text[j].isspace():
                j += 1
            if j < len(original_text):
                next_char = original_text[j]
                # If next_char is lowercase letter, probably a mistaken split (abbreviation)
                if next_char.islower():
                    ctx_prev = prev_sent[-60:]
                    ctx_next = original_text[j:j+60]
                    flags.append({
                        "type": "false_positive_lowercase_follow",
                        "prev_end_context": ctx_prev,
                        "next_start_context": ctx_next,
                        "prev_sentence_len": len(prev_sent.split()),
                        "next_sentence_len": len(next_sent.split())
                    })
        except ValueError:
            # couldn't locate, skip
            pass
        pos = max(0, end_prev - 5)
    return flags


def find_missed_splits(sbd_sentences):
    flags = []
    for s in sbd_sentences:
        wcount = len(s.split())
        if wcount > LONG_SENT_WORDS:
            flags.append({
                "type": "long_sentence_missed_split",
                "sentence_length_words": wcount,
                "snippet": " ".join(s.split()[:80]) + (" ..." if wcount > 80 else "")
            })
    return flags


def doc_diagnostics(doc_id, text, sbd_sentences):
    regex_sentences = simple_regex_split(text)
    diag = {
        "doc_id": doc_id,
        "num_chars": len(text),
        "num_tokens": len(re.findall(r"\w+", text)),
        "sbd_num_sentences": len(sbd_sentences),
        "regex_num_sentences": len(regex_sentences),
        "sbd_avg_sent_len_words": (sum(len(s.split()) for s in sbd_sentences) / max(1, len(sbd_sentences))),
        "regex_avg_sent_len_words": (sum(len(s.split()) for s in regex_sentences) / max(1, len(regex_sentences))),
        "sbd_sentences_sample": sbd_sentences[:6],
        "regex_sentences_sample": regex_sentences[:6],
        "flags": []
    }

    # Heuristic checks
    fp_flags = find_false_positive_splits(text, sbd_sentences)
    missed_flags = find_missed_splits(sbd_sentences)
    diag["flags"].extend(fp_flags)
    diag["flags"].extend(missed_flags)
    return diag


def sample_and_run(csv_path, out_path=OUT_JSONL, sample_size=SAMPLE_SIZE, seed=42):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    n = len(df)
    sample_size = min(sample_size, n)
    random.seed(seed)
    indices = random.sample(range(n), sample_size)

    diagnostics = []
    flagged_docs = 0
    for idx in indices:
        text = str(df.loc[idx, "Text"])
        doc_id = df.index[idx]
        try:
            sents = split_into_sentences(text)
        except Exception as e:
            sents = simple_regex_split(text)
        diag = doc_diagnostics(doc_id, text, sents)
        if diag["flags"]:
            flagged_docs += 1
        diagnostics.append(diag)

    # Save JSONL
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in diagnostics:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Print quick summary
    total = len(diagnostics)
    pct_flagged = (flagged_docs / total) * 100.0
    print(f"Sampled {total} documents from {csv_path}.")
    print(f"Flagged documents (heuristic): {flagged_docs} / {total}  ({pct_flagged:.1f}%)")
    # show top 10 example flags
    examples_shown = 0
    for d in diagnostics:
        if d["flags"] and examples_shown < 10:
            print("----")
            print(f"doc_id: {d['doc_id']}, sbd_sents: {d['sbd_num_sentences']}, regex_sents: {d['regex_num_sentences']}")
            print("Example SBD sample (first 3):")
            for s in d["sbd_sentences_sample"][:3]:
                print(" *", s[:200].replace("\n", " "))
            print("Flags (count):", len(d["flags"]))
            # print first flag for quick view
            print("First flag:", d["flags"][0])
            examples_shown += 1
    print("Diagnostics saved to:", out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/summary.csv", help="path to CSV with Text column")
    parser.add_argument("--out", type=str, default=OUT_JSONL, help="output jsonl diagnostics path")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE, help="number of documents to sample")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    sample_and_run(args.csv, out_path=args.out, sample_size=args.sample, seed=args.seed)

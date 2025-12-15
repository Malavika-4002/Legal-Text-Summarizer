import json
import numpy as np
from tqdm import tqdm

PATH = "data/processed/abstractive_dataset.jsonl"

input_lens = []
target_lens = []

with open(PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        obj = json.loads(line)
        input_lens.append(len(obj["input"].split()))
        target_lens.append(len(obj["target"].split()))

def stats(arr):
    arr = np.array(arr)
    return {
        "count": len(arr),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95))
    }

print("INPUT stats:", stats(input_lens))
print("TARGET stats:", stats(target_lens))

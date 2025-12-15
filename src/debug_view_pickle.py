""""
import pickle

path = "data/processed/roles_dev.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print("Total records:", len(data))
print("Sample item:\n")
print(data[0])
"""
import pickle
from collections import Counter

with open("data/processed/roles_train.pkl", "rb") as f:
    train_data = pickle.load(f)

all_labels = [label for doc in train_data for label in doc['labels']]
print(Counter(all_labels))

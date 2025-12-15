import json
import pickle

TRAIN_JSON = "data/raw/role_train1.json"
DEV_JSON = "data/raw/role-dev1.json"

OUT_TRAIN = "data/processed/roles_train.pkl"
OUT_DEV = "data/processed/roles_dev.pkl"

def load_rhetorical_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []

    for doc in data:
        annotations = doc.get("annotations", [])
        if not annotations:
            continue

        result_list = annotations[0].get("result", [])
        if not result_list:
            continue

        sentences = []
        labels = []

        for item in result_list:
            value = item.get("value", {})
            text = value.get("text", "").strip()
            label_list = value.get("labels", [])

            if not text or not label_list:
                continue

            # Many labels are like ["FAC"], ["PREAMBLE"]
            label = label_list[0].upper()

            sentences.append(text)
            labels.append(label)

        if sentences:
            processed.append({
                "sentences": sentences,
                "labels": labels
            })

    return processed


def main():
    print("Loading training data...")
    train_data = load_rhetorical_dataset(TRAIN_JSON)
    print("Training samples:", len(train_data))

    print("Loading dev data...")
    dev_data = load_rhetorical_dataset(DEV_JSON)
    print("Dev samples:", len(dev_data))

    print("Saving...")
    with open(OUT_TRAIN, "wb") as f:
        pickle.dump(train_data, f)

    with open(OUT_DEV, "wb") as f:
        pickle.dump(dev_data, f)

    print("DONE — Correct preprocessing completed!")


if __name__ == "__main__":
    main()

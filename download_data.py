import json
import os
import pandas as pd
from datasets import load_dataset

# 1. Define Paths
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
ROLE_PATH = os.path.join(RAW_DIR, "rhetorical_roles.json")
ILDC_PATH = os.path.join(RAW_DIR, "ILDC_single.csv")

def download_rhetorical_roles():
    if os.path.exists(ROLE_PATH):
        print(f"✅ Rhetorical Role data already exists at {ROLE_PATH}")
        return

    print("⏳ Downloading OpenNyAI Rhetorical Roles (Phase 2A)...")
    try:
        # CORRECTED ID: opennyaiorg/InRhetoricalRoles
        dataset = load_dataset("opennyaiorg/InRhetoricalRoles", split="train")
        
        data_to_save = []
        for row in dataset:
            # The structure is usually simple text/labels in this version
            if 'text' in row and 'labels' in row:
                # Some versions come sentence-split, some document-level
                # We will handle the standard format
                data_to_save.append({
                    'text': row['text'],
                    'label': row['labels'] 
                })
            # Check if it's the complex nested structure
            elif 'annotations' in row:
                 for item in row['annotations'][0]['result']:
                    data_to_save.append({
                        'text': item['value']['text'],
                        'label': item['value']['labels'][0]
                    })
        
        # Fallback: If empty, try the other split or structure
        if not data_to_save:
             print("⚠️ Warning: Structure changed. Saving raw dump to inspect.")
             data_to_save = [x for x in dataset]

        with open(ROLE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=None)
        print(f"✅ Saved {len(data_to_save)} sentences to {ROLE_PATH}")
        
    except Exception as e:
        print(f"❌ Error downloading Rhetorical Roles: {e}")

def download_ildc():
    if os.path.exists(ILDC_PATH):
        print(f"✅ ILDC Dataset already exists at {ILDC_PATH}")
        return

    print("⏳ Downloading ILDC Dataset (Phase 2B)...")
    try:
        # CORRECTED ID: trustworthy-nlp/ILDC is a stable mirror
        dataset = load_dataset("trustworthy-nlp/ILDC", split="train")
        
        # Convert to Pandas
        df = pd.DataFrame(dataset)
        
        # Save to CSV
        df.to_csv(ILDC_PATH, index=False)
        print(f"✅ Saved ILDC ({len(df)} cases) to {ILDC_PATH}")
        
    except Exception as e:
        print(f"❌ Error downloading ILDC: {e}")
        print("💡 MANUAL FIX: Go to https://github.com/Exploration-Lab/CJPE")
        print("   Download 'ILDC_single.csv.zip', unzip it, and put it in 'data/raw/'")

if __name__ == "__main__":
    print("🚀 Starting Data Download (Corrected IDs)...")
    download_rhetorical_roles()
    download_ildc()
    print("🎉 Done.")
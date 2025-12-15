import os
import requests
import zipfile

# --- CONFIGURATION ---
RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# List of ALL possible locations (The script will try them one by one)
ILDC_CANDIDATES = [
    # Candidate 1: Main branch, 'Data' folder (Most likely current)
    "https://github.com/Exploration-Lab/CJPE/raw/main/Data/ILDC_single.csv.zip",
    # Candidate 2: Master branch, 'Data' folder (Older)
    "https://github.com/Exploration-Lab/CJPE/raw/master/Data/ILDC_single.csv.zip",
    # Candidate 3: Main branch, 'Dataset' folder
    "https://github.com/Exploration-Lab/CJPE/raw/main/Dataset/ILDC_single.csv.zip",
]

RHETORICAL_CANDIDATES = [
    # Candidate 1: Legal-NLP-EkStep (Original Repo), 'main' branch
    "https://raw.githubusercontent.com/Legal-NLP-EkStep/rhetorical-role-baseline/main/data/train.json",
    # Candidate 2: Legal-NLP-EkStep, 'master' branch
    "https://raw.githubusercontent.com/Legal-NLP-EkStep/rhetorical-role-baseline/master/data/train.json",
    # Candidate 3: OpenNyAI Benchmarks (New Repo)
    "https://raw.githubusercontent.com/OpenNyAI/OpenNyAI-Benchmarks/main/LEGAL-PE/Rhetorical_Roles/data/train.json",
]

def try_download(candidates, save_path, is_zip=False):
    print(f"\n🔍 Searching for: {os.path.basename(save_path)}")
    
    for url in candidates:
        print(f"   Trying: {url} ...", end=" ")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, stream=True)
            
            if response.status_code == 200:
                print("✅ FOUND!")
                
                # Write to file
                temp_path = save_path + ".temp"
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # If it's a zip, extract it
                if is_zip:
                    print("   📦 Unzipping...")
                    try:
                        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                            zip_ref.extractall(RAW_DIR)
                        print(f"   🎉 Extracted to {RAW_DIR}")
                        os.remove(temp_path)
                        return True
                    except zipfile.BadZipFile:
                        print("   ❌ Downloaded file was not a valid ZIP. Trying next...")
                        os.remove(temp_path)
                        continue
                else:
                    os.replace(temp_path, save_path)
                    print(f"   🎉 Saved to {save_path}")
                    return True
            else:
                print(f"❌ {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    print(f"⚠️ ALL ATTEMPTS FAILED for {os.path.basename(save_path)}")
    return False

def check_existing_files():
    ildc_exists = os.path.exists(os.path.join(RAW_DIR, "ILDC_single.csv"))
    role_exists = os.path.exists(os.path.join(RAW_DIR, "rhetorical_roles.json"))
    
    if ildc_exists: print("✅ ILDC_single.csv is already present.")
    else: try_download(ILDC_CANDIDATES, os.path.join(RAW_DIR, "ILDC_single.csv"), is_zip=True)
    
    if role_exists: print("✅ rhetorical_roles.json is already present.")
    else: try_download(RHETORICAL_CANDIDATES, os.path.join(RAW_DIR, "rhetorical_roles.json"), is_zip=False)

if __name__ == "__main__":
    print("🚀 Starting Smart Download...")
    check_existing_files()
    print("\n👋 Done. Check your data/raw/ folder.")
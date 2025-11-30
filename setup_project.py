import os

# The Project Structure
structure = {
    "data/raw": [],
    "data/processed": [],
    "models/sbd_weights": [],
    "models/role_classifier": [],
    "models/t5_summarizer": [],
    "notebooks": ["01_data_exploration.ipynb"],
    "src": ["__init__.py", "config.py", "utils.py"],
    "src/sbd": ["__init__.py", "model_cnn.py", "postprocess.py"],
    "src/classifier": ["__init__.py", "dataset.py", "model_bilstm.py", "train.py"],
    "src/summarizer": ["__init__.py", "scorer.py", "t5_finetune.py"],
}

root_files = ["main.py", "requirements.txt", ".gitignore"]

def create_structure():
    print("🚧 Building Project Structure...")
    
    # Create Root Files
    for file in root_files:
        with open(file, 'w') as f:
            if file == ".gitignore":
                f.write("data/\nmodels/\n__pycache__/\n*.ipynb_checkpoints/\n.env")
            if file == "requirements.txt":
                f.write("torch\ntransformers\npandas\nnumpy\nscikit-learn\n")
            pass
        print(f"   Created: {file}")

    # Create Folders and Sub-files
    for folder, files in structure.items():
        os.makedirs(folder, exist_ok=True)
        print(f"   Created Folder: {folder}/")
        for file in files:
            filepath = os.path.join(folder, file)
            with open(filepath, 'w') as f:
                pass
            print(f"      Created File: {filepath}")

    print("\n✅ Project Structure Ready! You can delete this setup script now.")

if __name__ == "__main__":
    create_structure()
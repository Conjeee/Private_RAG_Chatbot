import os
from huggingface_hub import hf_hub_download

def download_model():
    # Configuration
    REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
    FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
    MODELS_DIR = "./models"
    
    # Check if file already exists
    file_path = os.path.join(MODELS_DIR, FILENAME)
    if os.path.exists(file_path):
        print(f"✅ Model found at: {file_path}")
        return

    print(f"⏳ Downloading {FILENAME} from HuggingFace... (This may take a while)")
    
    # Download logic
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("./models", exist_ok=True)
    download_model()
import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_models():
    # Base Model Directory
    MODELS_DIR = "./models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- 1. The Main Brain (Phi-3 GGUF) ---
    # This is a single file download
    print("\n⬇️  Phase 1: Downloading LLM (Phi-3)...")
    PHI_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
    PHI_FILE = "Phi-3-mini-4k-instruct-q4.gguf"
    phi_path = os.path.join(MODELS_DIR, PHI_FILE)
    
    if os.path.exists(phi_path):
        print(f"   ✅ Phi-3 found at: {phi_path}")
    else:
        try:
            print("   ⏳ Downloading Phi-3... (approx 2.4 GB)")
            hf_hub_download(
                repo_id=PHI_REPO,
                filename=PHI_FILE,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False
            )
            print("   ✅ Phi-3 Download complete!")
        except Exception as e:
            print(f"   ❌ Error downloading Phi-3: {e}")

    # --- 2. The Embedding Model (MiniLM) ---
    # This is a FOLDER download (snapshot)
    print("\n⬇️  Phase 2: Downloading Embedding Model (MiniLM)...")
    EMBED_REPO = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIR = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
    
    # Check if folder exists and is not empty
    if os.path.exists(EMBED_DIR) and os.listdir(EMBED_DIR):
        print(f"   ✅ Embedding model found at: {EMBED_DIR}")
    else:
        try:
            print("   ⏳ Downloading Embedding Model... (approx 80 MB)")
            snapshot_download(
                repo_id=EMBED_REPO,
                local_dir=EMBED_DIR,
                local_dir_use_symlinks=False  # Crucial for offline use
            )
            print("   ✅ Embedding Model Download complete!")
        except Exception as e:
            print(f"   ❌ Error downloading Embedding Model: {e}")

if __name__ == "__main__":
    download_models()
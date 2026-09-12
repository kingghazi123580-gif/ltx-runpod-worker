import os
from huggingface_hub import hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable set nahi hai!")
    exit(1)

try:
    print("Testing HF token access to Lightricks/LTX-2.5...")
    
    path = hf_hub_download(
        repo_id="Lightricks/LTX-2.5",
        filename="vae/ltx-2.5-audio-vae-bf16.safetensors",
        token=HF_TOKEN,
        local_dir="./test_download",
        force_download=False,
    )
    
    size = os.path.getsize(path)
    print(f"SUCCESS! File downloaded/accessible at: {path}")
    print(f"File size: {size} bytes")
    
except Exception as e:
    print(f"FAILED! Error: {type(e).__name__}")
    print(f"Message: {str(e)}")
    
    if "401" in str(e):
        print("\n>>> 401 Unauthorized: Token invalid ya missing hai.")
    elif "403" in str(e):
        print("\n>>> 403 Forbidden: Token valid hai LEKIN license accept nahi kiya.")
        print(">>> Browser mein https://huggingface.co/Lightricks/LTX-2.5 par jayein aur license accept karein.")
    elif "gated" in str(e).lower():
        print("\n>>> Gated repo access issue. License accept karein.")
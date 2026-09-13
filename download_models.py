import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "/app/models/ltx-2.5"
os.makedirs(MODEL_DIR, exist_ok=True)

# Yeh file build ke doran download hogi — RunPod ke servers par
REQUIRED_FILES = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
]

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN build argument missing")

for f in REQUIRED_FILES:
    print(f"Downloading: {f}")
    hf_hub_download(
        repo_id="Lightricks/LTX-2.5",
        filename=f,
        token=HF_TOKEN,
        local_dir=MODEL_DIR,
    )
    print(f"Downloaded: {f}")

print("All models downloaded successfully!")

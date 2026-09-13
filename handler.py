import runpod
import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "/app/models/ltx-2.5"

REQUIRED_FILES = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
]

def ensure_model_downloaded():
    hf_token = os.environ.get("HF_TOKEN")
    status = {}
    for f in REQUIRED_FILES:
        local_path = os.path.join(MODEL_DIR, f)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
            status[f] = "already_cached"
        else:
            hf_hub_download(
                repo_id="Lightricks/LTX-2.5",
                filename=f,
                token=hf_token,
                local_dir=MODEL_DIR,
            )
            status[f] = "downloaded"
    return status

def handler(event):
    try:
        result = ensure_model_downloaded()
        return {"status": "success", "stage": "model_download", "files": result}
    except Exception as e:
        import traceback
        return {"status": "error", "stage": "model_download", "message": str(e),
                 "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})

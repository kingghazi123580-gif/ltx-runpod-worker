import runpod
import os
from huggingface_hub import hf_hub_download

def handler(event):
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return {"status": "error", "stage": "auth_check", "message": "HF_TOKEN not set"}

        # Sabse chhoti file se test karo (~348MB)
        path = hf_hub_download(
            repo_id="Lightricks/LTX-2.5",
            filename="vae/ltx-2.5-audio-vae-bf16.safetensors",
            token=hf_token,
        )
        size = os.path.getsize(path)
        return {"status": "success", "downloaded_path": path, "size_bytes": size}
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "stage": "model_download",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

runpod.serverless.start({"handler": handler})

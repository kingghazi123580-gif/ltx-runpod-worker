import runpod
import os
import torch
from diffusers import AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio

MODEL_DIR = "/runpod-volume/models/ltx-2.5"

def handler(event):
    try:
        import traceback
        results = {}

        # Video VAE load karo
        video_vae_path = os.path.join(MODEL_DIR, "vae/ltx-2.5-video-vae-bf16.safetensors")
        video_vae = AutoencoderKLLTX2Video.from_single_file(
            video_vae_path,
            torch_dtype=torch.bfloat16,
        )
        results["video_vae_loaded"] = True
        results["video_vae_class"] = video_vae.__class__.__name__

        # Audio VAE load karo
        audio_vae_path = os.path.join(MODEL_DIR, "vae/ltx-2.5-audio-vae-bf16.safetensors")
        audio_vae = AutoencoderKLLTX2Audio.from_single_file(
            audio_vae_path,
            torch_dtype=torch.bfloat16,
        )
        results["audio_vae_loaded"] = True
        results["audio_vae_class"] = audio_vae.__class__.__name__

        return {"status": "success", "stage": "vae_load", "results": results}

    except Exception as e:
        return {"status": "error", "stage": "vae_load", "message": str(e),
                "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})

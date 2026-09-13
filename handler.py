import runpod
import os
import torch
from diffusers import AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
from transformers import GemmaTokenizer, Gemma3ForConditionalGeneration

MODEL_DIR = "/runpod-volume/models/ltx-2.5"

def handler(event):
    try:
        import traceback
        results = {}

        # 1. Video VAE load test
        vae_path = os.path.join(MODEL_DIR, "vae/ltx-2.5-video-vae-bf16.safetensors")
        if os.path.exists(vae_path):
            results["video_vae_exists"] = True
            results["video_vae_size"] = os.path.getsize(vae_path)
        else:
            results["video_vae_exists"] = False

        # 2. Audio VAE load test
        audio_vae_path = os.path.join(MODEL_DIR, "vae/ltx-2.5-audio-vae-bf16.safetensors")
        if os.path.exists(audio_vae_path):
            results["audio_vae_exists"] = True
            results["audio_vae_size"] = os.path.getsize(audio_vae_path)
        else:
            results["audio_vae_exists"] = False

        # 3. Text encoder file check
        text_enc_path = os.path.join(MODEL_DIR, "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors")
        if os.path.exists(text_enc_path):
            results["text_encoder_exists"] = True
            results["text_encoder_size"] = os.path.getsize(text_enc_path)
        else:
            results["text_encoder_exists"] = False

        # 4. Transformer file check
        transformer_path = os.path.join(MODEL_DIR, "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
        if os.path.exists(transformer_path):
            results["transformer_exists"] = True
            results["transformer_size"] = os.path.getsize(transformer_path)
        else:
            results["transformer_exists"] = False

        return {"status": "success", "stage": "model_files_check", "files": results}

    except Exception as e:
        return {"status": "error", "stage": "model_check", "message": str(e),
                "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})

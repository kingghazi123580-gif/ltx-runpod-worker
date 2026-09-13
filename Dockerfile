FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Har model file ke liye alag RUN — taake har layer 10GB se kam rahe
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-2.5', filename='vae/ltx-2.5-audio-vae-bf16.safetensors', token=os.environ['HF_TOKEN'], local_dir='/app/models/ltx-2.5')"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-2.5', filename='vae/ltx-2.5-video-vae-bf16.safetensors', token=os.environ['HF_TOKEN'], local_dir='/app/models/ltx-2.5')"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-2.5', filename='text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors', token=os.environ['HF_TOKEN'], local_dir='/app/models/ltx-2.5')"
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Lightricks/LTX-2.5', filename='diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors', token=os.environ['HF_TOKEN'], local_dir='/app/models/ltx-2.5')"

COPY handler.py .
CMD ["python3", "-u", "handler.py"]

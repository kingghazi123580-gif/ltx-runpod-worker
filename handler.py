import runpod
import subprocess

def handler(event):
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return {"status": "success", "gpu_info": result.stdout}
    except Exception as e:
        return {"status": "error", "stage": "gpu_check", "message": str(e)}

runpod.serverless.start({"handler": handler})

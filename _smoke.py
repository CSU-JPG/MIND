import torch
import lpips
import torchmetrics
import pyiqa
import clip
import transformers
import av
import huggingface_hub

print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
print(f"huggingface_hub={huggingface_hub.__version__}")
print(f"pyiqa={pyiqa.__version__}")
print(f"lpips={lpips.__version__}")
print(f"clip OK; torchmetrics OK; av OK")

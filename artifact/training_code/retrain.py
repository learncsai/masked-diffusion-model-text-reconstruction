"""Run from training_code/ only after restoring the hashed training inputs."""
import os
import sys
from pathlib import Path

root=Path(__file__).resolve().parent
if not (root/'src/diffusion_lm_rmt/matched_mdlm.py').exists():
    raise SystemExit('Place this wrapper in training_code/ before using it.')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.pop('MDLM_TRAIN_DEADLINE', None)
sys.path[:0]=[str(root/'src'),str(root/'runpod'),str(root/'colab')]
import torch
from runtime import configurations
from diffusion_lm_rmt import matched_mdlm as core
if not torch.cuda.is_available():raise SystemExit('A CUDA GPU is required for this full training recipe.')
torch.set_num_threads(min(6,os.cpu_count() or 1))
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
for config in configurations(root,'reviewer_retrain_v1','full'):
    core.run(root,config,torch.device('cuda'))

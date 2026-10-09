"""Train matched MDLMs locally with the recorded three-corpus implementation."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'runpod'), str(ROOT/'colab')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='mdlm_runpod_main_v1')
    parser.add_argument('--mode', choices=['full','smoke'], default='full')
    parser.add_argument('--corpus', choices=['tinystories','wikitext','cnn_dailymail'])
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from runtime import configurations, render_results
    from diffusion_lm_rmt import matched_mdlm
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required. Offline results: python reproduce.py')
    torch.set_num_threads(min(6,os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for config in configurations(ROOT,args.label,args.mode):
        if args.corpus and config['corpus'] != args.corpus:
            continue
        matched_mdlm.run(ROOT,config,torch.device('cuda'))
    render_results(ROOT,args.label,args.mode)


if __name__ == '__main__':
    main()

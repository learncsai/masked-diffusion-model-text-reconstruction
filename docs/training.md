# Matched masked diffusion model training

This protocol follows Section 5 and Appendix C.1-C.2 of the submitted paper. The offline path rebuilds reported neural summaries from saved measurements; it does not run inference or training.

## Architecture and optimization

| Setting | Recorded value |
|---|---|
| Transformer | Bidirectional, two blocks, hidden width 96 |
| Attention / feed-forward | Four heads / width 384 |
| Embeddings / dropout | Tied input/output / zero |
| Parameters | 5,179,539 |
| Output vocabulary | 50,257 classes, including EOS; excludes MASK/PAD |
| Input length | 64 tokens |
| Corruption time | Uniform on [0.05,0.95]; absorbing independent token masks |
| Auxiliary parent | 4,000 updates per corpus |
| A/B fits | 12,000 updates; evaluated at 8,000 and 12,000 |
| Batch | Four microbatches of four; effective batch 16 |
| AdamW | beta=(0.9,0.98), weight decay 0.01, gradient clip 1 |
| Learning rate | Warm up to 5e-4 in 400 updates; decrease to 5e-5 at 8,000; remain there to 12,000 |
| Arithmetic | float32; deterministic algorithms; TF32/cuDNN benchmarking disabled |
| Evaluation | Same 128 chunks and saved q=t=0.5 masks for counts and MDLM |

Each corpus has one auxiliary parent, 18 A/B fits (three pairs x three sizes x two arms), and three alternate-training-seed controls: **22 runs per corpus, 66 total**. A/B fits start from a shared parent with fresh optimizer state, then share minibatch indices, masks, and diffusion times. Auxiliary initialization and training randomness are conditioned on when comparing corpora.

## Recorded runtime and implementation

The archived run used Linux, Python 3.12, PyTorch 2.8.0+cu128, CUDA 12.8, and an RTX 4090 with 24 GB. Logged training totals **7.91 hours**, excluding preparation and evaluation. This describes the recorded run; it is not a runtime guarantee.

The exact three-corpus implementation is preserved at `artifact/training_code/src/diffusion_lm_rmt/matched_mdlm.py`. Its SHA-256 is `fb49f155876e332ebc4521cf5eccb423607c7b7e3d52e63db5bbe9e59e16a5b6`. The root package uses this implementation with its runtime-helper import adapted for package installation and new manifests recording this repository's Git revision. The original workspace's TinyStories-only loader is not used by this repository's training entry point.

## New local training run

Prepare all corpora with `scripts/prepare_public_data.py`, install the [CUDA experiment environment](setup.md), then:

```bash
python scripts/train_mdlm.py
```

The default writes new models to `results/mdlm_runpod_main_v1/<corpus>/`, the layout expected by downstream evaluation scripts. It trains on locally prepared inputs and does not call a cloud API. Use `--corpus tinystories` to train one corpus. All three are needed for the full reported grid.

For a small execution check on the existing prepared inputs:

```bash
python scripts/train_mdlm.py --mode smoke --label smoke_local --corpus tinystories
```

The smoke recipe changes sizes, chunks, and update counts and does not reproduce paper results. Changing frozen code/configuration/inputs requires a new label or fresh working copy. A new run can checkpoint and resume its own trajectory. Historical optimizer/random-state files are unavailable, so exact continuation of the original run cannot be claimed.

## Quality, rankings, and directional comparison

After full training on all three corpora:

```bash
python scripts/freeze_matched_inputs.py
python scripts/evaluate_lag_disagreement.py --prepare-inputs
python scripts/evaluate_lag_disagreement.py
python scripts/evaluate_lag_topk.py
python scripts/evaluate_mdlm_topk.py
python scripts/validate_mdlm_topk.py
python scripts/build_mdlm_paper_results.py
python scripts/evaluate_full_context_linear.py --stage fit
python scripts/evaluate_mdlm_subspaces.py
```

The input freeze creates an ignored local ZIP containing the new run's input identity and count context, which the historical top-k evaluator expects. Count ranking uses all 50,257 raw scores; neural ranking uses logits. Ties use ascending token ID. MSE and top-k results average within hidden positions and then equally across evaluation chunks.

The directional test uses saved source folds, a 64-dimensional vocabulary readout, and rank-4 directions selected on the opposite fold. It compares predicted directions, empirical directions, and the shared unigram/frequency control. Random rank-4 capture in dimension 64 has expectation 6.25%. Input ranking and directional overlap answer separate questions from aggregate calibration.

Some evaluation scripts embed the full-run path `mdlm_runpod_main_v1`; keep the default label for this recipe. Custom labels require adapting those paths consistently in a new working copy. Neural checkpoint comparison will fail if architecture or hashed inputs differ from the training manifest.

## Historical input recovery

The original per-corpus manifests in component 06 identify 58 unique training-input paths and hashes. Full token arrays and 129 model snapshots are excluded from the compact release. [Archived TRAINING.md](../artifact/TRAINING.md) lists those prerequisites and includes `artifact/training_code/retrain.py`, intended for an independently restored historical input export. Fresh public-data preparation uses new allocations and does not claim to match that export.

# MDLM training and full reruns

## Recorded experiment

The 66 runs used one NVIDIA GeForce RTX 4090 (24 GB) on RunPod, Linux,
Python 3.12.3, PyTorch 2.8.0+cu128 and CUDA 12.8. Each of the three corpora
has one auxiliary model, 18 A/B models, and three alternate-seed controls.
The auxiliary model receives 4,000 updates. The other models receive 12,000
updates with evaluation at 8,000 and 12,000. Summed elapsed training times
from the logs total 7.91 hours. This excludes preparation and evaluation and
is not a promised runtime on another machine.

The model has two Transformer blocks, width 96, four heads, feed-forward width
384, shared input/output embeddings, no dropout, and 5,179,539 parameters.
Training uses float32 and effective batch size 16 through four microbatches
of four. Appendix C gives the complete objective, time interval, optimizer,
schedule and seed controls. `training_records/runtime_environment.json` gives
the recorded runtime. `benchmark.json` is a planning estimate, not elapsed time.

## Exact code version

`training_code/` contains the archived three-corpus RunPod implementation.
Its `src/diffusion_lm_rmt/matched_mdlm.py` SHA-256 is
`fb49f155876e332ebc4521cf5eccb423607c7b7e3d52e63db5bbe9e59e16a5b6`.
This is important because an earlier local loader handled only TinyStories.
No cloud billing or pod-stop launcher needs to be run to reproduce training.

## Inputs required for training

The complete training inputs are not part of this compact results package.
The per-corpus `manifest.json` files in component 06 list the exact 58 unique
input paths and their hashes. Relative to the training directory, they include:

- `data/diffusion_llm/processed_64_<corpus>/`: `replicate3_train_a.pt`, `eval.pt`,
  and `confirmatory_split_manifest.json`. The three directory suffixes are
  `tinystories_matched`, `wikitext_stats`, and `cnn_dailymail_capped`.
- `data/diffusion_llm/reproducibility_followup_v1/<corpus>/`:
  `raw_documents.jsonl`, `pair{1,2,3}_{a,b}.npz` and their source/chunk JSON records.
- `data/diffusion_llm/calibration_diagnostics_v1/<corpus>/`:
  `grid_512_512_q0.5.npz`, `reference_original.npz`, `reference_original.json`.
- `data/diffusion_llm/gpt2_tokenizer/tokenizer_metadata.json`.

Use the original hashed input export or regenerate the input allocations from
the public sources with the documented preprocessing and check every hash.
The full RunPod execution export is retained separately. It is not inside this
compact submission archive. A changed public dataset revision may prevent exact
recovery of an old hash. Such a rerun must be described as a new replication.
`NEWS_DATA_ACCESS.md` documents the news preprocessing and retained rosters.
This submission excludes text and reversible token sequences from every corpus.
DATA_PREPARATION.md describes the complete source/chunk workflow.

## Optional training command

With those files placed under `training_code/`, install a CUDA-enabled PyTorch
build matching the recorded runtime and the NumPy/SciPy requirements. Then run:

```sh
cd training_code
python retrain.py
```

The supplied wrapper uses the archived scientific configuration and a new output
label, `reviewer_retrain_v1`. It does not call the RunPod API. Training was not
repeated during artifact validation. It requires several GPU hours.

Saved local weights comprise 129 snapshots. The reviewer ZIP omits those weights
and optimizer state. Optimizer/RNG continuation files were not downloaded from
the original Pod. The code supports checkpointing in a new run, but exact
continuation of the historical training trajectory cannot be promised.
Different GPU kernels or software versions can change floating-point results.

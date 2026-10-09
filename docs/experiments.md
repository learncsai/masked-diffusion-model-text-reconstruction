# Experiment recipes

Run commands from the repository root with the [experiment environment](setup.md). The default offline reproduction covers the reported outputs without rerunning expensive experiments. Commands below create new scientific measurements and store them under ignored `data/` and `results/`.

## Overview

| Study | Paper | Configuration | Sampling design |
|---|---|---|---|
| Main finite-reference correction | Sections 3-4; B.1-B.2 | `rare_context_prospective_v1.json` | 20 A/B pairs, six banks per corpus; n=512/1024/2048/4096; m=1024/2048/4096, plus 8192 baseline |
| Additional baselines and confirmation | B.2-B.3 | `correction_followup_v1.json` | Same original grid; six fresh pairs and three fresh banks, n,m=1024/2048 |
| Added confirmation bootstrap | B.3, Table 6 | `bootstrap_confirmation_v1.json` | Existing confirmation allocations at m=2048,n=1024 |
| Multiplier transfer | B.3, Figure 3 | `oracle_multiplier_v1.json` | Fit factors on named original cells/corpora, evaluate held-out records |
| Matched count/MDLM | Section 5; C-D | `matched_mdlm_tinystories_pilot_v1.json` | Three pairs; n=512/1024/2048; q=0.5; shared 128 evaluation chunks |
| Directional overlap | C.2 | Runner settings | Two source folds; rank 4; dimension 64; frequency control |
| Synthetic decomposition and oracles | E.1-E.2 | `rare_context_development_v1.json` | 80 pairs; iid, dependent, four-chunk sources; n=256/1024 |
| Smoothing and rare contexts | Table 3; E.3 | `alpha_sweep_v1.json` | Same 20 pairs/six banks; alpha=0.5/1/2/5/10/20/50; fixed/refit weights |

Seeds that resemble calendar dates are fixed integers from the archived configurations. They do not indicate dates for a new run. New preparations and output hashes must be reported as new allocations. The historical timing/known-outcome status is described in [STUDIES.md](../artifact/STUDIES.md).

## 1. Synthetic controls and the development gate

These require no public corpus:

```bash
python scripts/run_rare_context_development.py
python scripts/run_rare_context_development.py --config configs/rare_context_alpha1_matched_v1.json
python scripts/run_rare_context_development.py --config configs/rare_context_alpha20_matched_v1.json
python scripts/diagnose_missing_contexts.py
python scripts/run_population_oracle.py
```

The first command runs all five development cases and saves nonlinear A/B outcomes, linearized comparisons, finite-reference forecasts, and coverage. The two additional configurations match alpha=1/20 to the clustered alpha=5 source draws, as required by Table 9's oracle comparison. The decomposition retains unseen-row energy, cross terms, seen-row estimation error, and propagation error. The oracle replaces reference estimates with exact population moments while retaining the same observed pairs.

Run the **full** development configuration before initializing the main correction study. `--pilot` is a small software check and is not the reported 80-pair study. The main runner checks the three alpha=5 cases for lower small-reference corrected error and positive scalar forecasts. Archive replay recalculates oracle comparisons from saved outcomes; the commands above resample the synthetic populations anew.

## 2. Twenty-pair correction study

First prepare all three corpora using [the public-data commands](data.md) and complete the synthetic gate. Then:

```bash
python scripts/run_rare_context_study.py initialize
python scripts/run_rare_context_study.py forecast --corpus tinystories
python scripts/run_rare_context_study.py forecast --corpus wikitext
python scripts/run_rare_context_study.py forecast --corpus cnn_dailymail
python scripts/run_rare_context_study.py freeze
python scripts/run_rare_context_study.py score --corpus tinystories
python scripts/run_rare_context_study.py score --corpus wikitext
python scripts/run_rare_context_study.py score --corpus cnn_dailymail
python scripts/run_rare_context_study.py summarize
```

The runner saves method/configuration hashes, freezes reference-only forecasts before scoring A/B outcomes, and reports source, source jackknife, and multinomial comparisons. The reference bank holds a nested 8,192-chunk allocation; the three smaller sizes feed the correction grid. The source-derived decision target is saved before scoring and produces supporting size-choice results.

`scripts/prepare_rare_context_study.py` is the **historical extension** allocator: it expects earlier source pools, manifests, and exclusions. Use `prepare_public_data.py` for a fresh clone. Do not bypass historical hash checks to force new inputs into an old manifest.

## 3. Additional baselines and fresh confirmation

After the main input preparation, run:

```bash
python scripts/run_correction_followup.py initialize
python scripts/check_multinomial_jackknife.py
python scripts/run_correction_followup.py baselines --corpus tinystories
python scripts/run_correction_followup.py baselines --corpus wikitext
python scripts/run_correction_followup.py baselines --corpus cnn_dailymail
python scripts/run_correction_followup.py prepare --corpus tinystories
python scripts/run_correction_followup.py prepare --corpus wikitext
python scripts/run_correction_followup.py prepare --corpus cnn_dailymail
python scripts/run_correction_followup.py forecast --corpus tinystories
python scripts/run_correction_followup.py forecast --corpus wikitext
python scripts/run_correction_followup.py forecast --corpus cnn_dailymail
python scripts/run_correction_followup.py freeze
python scripts/run_correction_followup.py score --corpus tinystories
python scripts/run_correction_followup.py score --corpus wikitext
python scripts/run_correction_followup.py score --corpus cnn_dailymail
python scripts/summarize_correction_followup.py
python scripts/summarize_correction_followup.py --confirmation
```

This adds nonlinear source bootstrap, bootstrap jackknife, direct-split `1/n` and constant forecasts, and fixed 1.25 inflation. Confirmation allocates unused source texts into six A/B pairs and three reference banks per corpus. Its designated primary comparison is corrected versus plain multinomial at m=1024,n=2048. New confirmation forecasts are frozen before scoring.

The full-grid intervals retain paired methods and nested sizes. Conditional correction uses the jackknife only for m<=n. Tables 2 and 5 compare these choices; Table 6 includes confirmation. The archived bootstrap extension was added after confirmation outcomes were known, and is not labeled a new prospective trial.

## 4. Bootstrap confirmation and multiplier transfer

```bash
python scripts/run_bootstrap_confirmation.py initialize
python scripts/run_bootstrap_confirmation.py run --corpus tinystories
python scripts/run_bootstrap_confirmation.py run --corpus wikitext
python scripts/run_bootstrap_confirmation.py run --corpus cnn_dailymail
python scripts/analyze_multiplier_transfer.py
python scripts/build_oracle_multiplier_results.py
python scripts/summarize_bootstrap_confirmation.py
```

This bootstrap calculation uses only the larger-reference confirmation cell m=2048,n=1024. All banks/corpora are retained; uncomputed bootstrap primary/full-grid entries remain explicitly "Not run". Multiplier transfer consumes completed original/confirmation results and synthetic oracle outputs. Its fitted factors use original training records, then evaluate held-out size cells or corpora. These retrospective comparisons must keep that status in a new report.

## 5. Smoothing sweep

Using the main twenty-pair allocations:

```bash
python scripts/run_alpha_sweep.py prepare
python scripts/run_alpha_sweep.py forecast
python scripts/run_alpha_sweep.py freeze
python scripts/run_alpha_sweep.py score
python scripts/summarize_alpha_sweep.py
```

The runner defaults to all configured corpora; `--corpus` can restrict preparation/forecast/scoring, and `--bank` can restrict forecasting. Freeze only after the required forecasts exist. Both fixed alpha=5 weights and auxiliary-refit weights are evaluated. Bootstrap is limited to alpha=1/5/20. The compact Table 3 uses the three m<n cells, fixed weights, and source-split-corrected source/multinomial laws. Appendix E.3 also uses query coverage and paired changes in size-response slopes; the full sweep is supporting evidence.

## 6. MDLM, quality, rankings, and directions

Follow [the training guide](training.md). It includes local training, input freezing for count top-k, neural rescoring, directional tests, and summaries. No RunPod account/API is needed to execute the supplied training implementation on an existing CUDA machine.

## Earlier studies in the paper

Appendix B.5 and D.1 retain original mask/size/shuffle/reference comparisons that use auxiliary partitions different from the matched correction task. Their compact measurements and retained calculation scripts are in `artifact/earlier_results/`, components 01 and 05, and the archive's rebuild scripts. `python reproduce.py` checks the retained records and outputs used by the submitted inventory. [The archived study guide](../artifact/STUDIES.md) documents exact counts and reuse.

The corresponding historical commands include `run_reproducibility_followup.py`, `prepare_calibration_diagnostics.py`, `prepare_lag_disagreement_extension.py`, `prepare_lag_reference_sensitivity.py`, and `run_reference_ratio.py`. They require the original earlier-stage processed splits and manifests listed in their configurations. The new matched preparation does not manufacture those historical partitions. Repeating those exact studies needs the matching input export; adapting them to freshly allocated inputs is a new replication.

Do not treat diagnostic sweeps, multiple reference banks on common outcomes, or reused pairs as new independent corpus observations. No image-diffusion training is part of this repository's reported experiments; Table 7 is a conceptual comparison.

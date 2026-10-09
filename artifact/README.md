# Reviewer guide: Predicting Corpus-to-Corpus Variation in Text Reconstruction

This supplement reproduces the numerical results in the **main text and
Appendices A through E**, including **Table 3's smoothing ablation**. It contains
code, configurations, compact numerical measurements and provenance. It includes
no corpus text, token sequences, target-token ranks, neural checkpoints or
manuscript. The archive is checked against a **20 MB** limit.

## Start here

Use Python 3.12 or later. The release was tested with Python 3.14.2 and the
versions in `requirements.txt`. From the extracted folder, run:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

After dependency installation, the script runs on CPU without downloads or a
GPU. It rebuilds **eight empirical tables and five submission figures**, checks
their values against manuscript fixtures, and supplies the two non-empirical
tables of definitions. Allow several minutes and 500 MB of working space.

Open **`reproduced/INDEX.md`** for links to every table and figure in submission
order. `reproduced/REPORT.json` records the checks and their scope.

## Table 3 and Appendix E

- **`reproduced/tables/table_3.csv`**: the 18 smoothing-ablation values.
- **`reproduced/tables/tab_main-alpha-ablation.tex`**: Table 3's LaTeX rows.
- **`reproduced/APPENDIX_E_CHECKS.json`**: independent Table 3 calculation,
  rare-context percentages, the retained synthetic decomposition, and the
  paired smoothing checks described in E.3.
- **`APPENDIX_E.md`**: a short guide to E.1 through E.4 and their evidence.
- **`ALPHA_SWEEP.md`**: grid, sampling units, aggregation and retained records.

Table 3 is recalculated from full/half reference forecasts and A/B pair scores,
rather than copied from the paper. It uses fixed alpha=5 weights and the three
m<n conditions in E.3. This calculation also checks agreement with an
independently implemented full-sweep summary.

Appendix F is **not part of the submission inventory**. Its full numerical
sweep records remain available to inspect unselected settings. Extra sweep
plots and tables go to `reproduced/diagnostics/`, apart from the five paper
figures in `reproduced/figures/`.

## Navigate the package

| File or folder | Purpose |
|---|---|
| ARTIFACT_MAP.md | Every submitted figure/table, with inputs and generator |
| APPENDIX_E.md | Retained synthetic controls, alpha ablation and reproduction claim |
| REPRODUCIBILITY.md | Offline checks and the limits of numerical replay |
| DATA_PREPARATION.md | Dataset access, source allocation, chunking, masks and rerun steps |
| STUDIES.md | Pair counts, reference banks, auxiliary splits and study timing |
| TRAINING.md | Recorded RTX 4090 training and requirements for a new training run |
| DATA_SOURCES.md, NEWS_DATA_ACCESS.md, LICENSE_REVIEW.txt | Attribution and published access terms |
| components/ | Nine numerical components used by the rebuild |
| research_code/, training_code/ | Statistical/preprocessing code and archived neural implementation |
| training_records/ | All 66 training logs and runtime records |

`reproduced/logs/` contains calculation logs. Each invocation creates a new
`reproduced/work_<timestamp>/` copy and preserves supplied evidence.
`python reproduce.py --extract-only` verifies hashes and makes a working copy
without analyses. `PACKAGE_SHA256.json` covers delivered files,
`REDUCTIONS.json` documents numerical reductions, and `RELEASE_SCOPE.json`
identifies this submission's inventory.

## What can be reproduced offline?

The default script recomputes aggregation, paired inference, source-split
corrections, size decisions, exact synthetic population moments and plots.
Top-k uses per-chunk hit rates. Directional comparisons use the reported
covariance blocks. Synthetic decomposition checks use saved component
summaries. Historical GPU rescoring and full-input audits are included as
records, rather than silently claimed as new runs.

Repeating tokenization, new real-text forecasts, neural inference or training
requires separately obtained inputs. Complete historical source allocations
and all position arrays are not delivered, and historical downloads were not
all pinned to immutable revisions. `DATA_PREPARATION.md` explains reconstruction
and hash checks. This is an offline **numerical reproduction**, not a
one-command reconstruction of the entire experiment from a fresh public
download. No new training is needed to rebuild the reported outputs.

Historical component READMEs refer to earlier exports. This guide and
`ARTIFACT_MAP.md` define the current reviewer release.

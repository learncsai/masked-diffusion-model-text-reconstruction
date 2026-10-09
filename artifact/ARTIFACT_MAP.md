# Submission map: main text and Appendices A through E

Run `python reproduce.py` at the archive root, then open `reproduced/INDEX.md`.
Numbers below follow the submitted document without Appendix F. Stable LaTeX
labels are also listed. Component paths start inside `components/`.

## Five figures

| Figure | Label | Measurements and generator |
|---|---|---|
| 1 | fig:reference-correction | 09_reference_correction: bank forecasts and 20 A/B scores. Root rebuild_correction.py and plot_correction.py |
| 2 | fig:mdlm-disagreement | 06_mdlm: per-chunk scores and matched count forecasts. scripts/build_mdlm_paper_results.py and scripts/plot_mdlm_paper_results.py |
| 3 | fig:correction-all | Component 09, same records and scripts, full size/reference grid |
| 4 | fig:multiplier-transfer | 11_oracle_multiplier: forecasts and pair means. scripts/analyze_multiplier_transfer.py and scripts/build_oracle_multiplier_results.py |
| 5 | fig:lag-topk | 05_topk: per-chunk hit rates. Root rebuild_lightweight.py and plot_topk.py |

These PDFs appear in `reproduced/figures/`. Additional diagnostic plots are in
`reproduced/diagnostics/` and are not additional submission figures.

## Ten tables

| Table | Label | Evidence and reconstruction |
|---|---|---|
| 1 | tab:data-roles | Protocol definition, not computed evidence |
| 2 | tab:correction-primary | 10_correction_followup: primary/full-grid errors for ten methods |
| 3 | tab:main-alpha-ablation | 13_alpha_sweep: full/half reference estimates and 20 pair scores. Root rebuild_appendix_e.py independently recomputes corrections and all 18 values, then checks the full-sweep summary |
| 4 | tab:main-mdlm | 06_mdlm: regenerated summary.csv, n=2048, lag and 12,000-update MDLM |
| 5 | tab:followup-intervals | Component 10: paired intervals and conditional policies |
| 6 | tab:confirmation | 12_bootstrap_confirmation: confirmation plus multiplier results from 11 and the added m>n bootstrap comparison |
| 7 | tab:correction-decisions | Component 10: targets, choices, failures, abstentions and excess size |
| 8 | tab:image-text-map | Conceptual correspondence from definitions and the cited image study |
| 9 | tab:mdlm-quality | Component 06: MSE and top-k rows regenerated from measurements |
| 10 | tab:population-oracle | Component 11: exact population moments and intervals from saved nonlinear pair outcomes |

The eight empirical tables are recalculated and reconciled against
`EXPECTED_RESULTS.json`. That file supplies comparison fixtures, not the
calculated measurements. The two conceptual/protocol tables are copied as
definitions and clearly marked. Table outputs are LaTeX rows, not full
manuscript layouts. Table 3 also has a CSV with explicit column names.

## Appendix E and supporting evidence

`APPENDIX_E.md` maps E.1 through E.4 to the files and checks. Table 10 retains
synthetic alpha=1 and alpha=20 cases. E.3's real-text alpha findings use component
13. The full seven-value sweep, both weight arms and all reference sizes remain
inspectable even though Appendix F is omitted.

Component 01 retains unordered queried-count histograms for the two coverage
percentages in E.3. Component 07 retains the exact two-fold covariance blocks
for the reported rank-4, dimension-64 directional test. Extra settings and
older studies appear under diagnostics/ and earlier_results/.

`STUDIES.md` describes allocation and reuse. `DATA_PREPARATION.md` explains the
source/chunk protocol and what a new experiment requires. `TRAINING.md` documents
the GPU runtime. Historical implementation checks are in
`docs/historical_audits/`. They are not replayed when full inputs or checkpoints
are absent.

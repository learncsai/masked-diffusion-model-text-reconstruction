# Guide to Appendix E

The submission retains a compact Appendix E and omits Appendix F. Run
`python reproduce.py` once for all offline calculations below.

| Section | Evidence | What the command does |
|---|---|---|
| E.1: controlled population and error decomposition | 09_reference_correction/results/rare_context_development_v1/missing_row_decomposition/summary.json | Checks the saved unseen-row/cross-term identity and reproduces the -32.6, -38.6 and +3.2 percentage-point example. It does not rerun the nonlinear simulation |
| E.2: population forecasts and smoothing | 11_oracle_multiplier | Recomputes exact population moments, checks small-population enumeration, and rebuilds all ten Table 10 rows and intervals using saved nonlinear pair scores |
| E.3: real-text smoothing and rare contexts | 13_alpha_sweep and 01_original_counts | Rebuilds all 18 Table 3 values, checks the weight-refit ranking and six paired slope-change intervals, and recomputes WikiText's 7.3% unseen and 20.8% below-five-count rates from unordered query histograms |
| E.4: reproduction and computation | training_records/, TRAINING.md, docs/historical_audits/ | Sums logged training time over all 66 runs. Preserves historical implementation and checkpoint-rescoring checks as records. Documents omitted inputs and acquisition requirements |

## Direct outputs

- `reproduced/tables/table_3.csv`
- `reproduced/tables/tab_main-alpha-ablation.tex`
- `reproduced/tables/tab_population-oracle.tex`
- `reproduced/APPENDIX_E_CHECKS.json`
- `reproduced/PROSE_CLAIMS.json`
- `reproduced/REPORT.json`

Table 3 uses **fixed reconstruction weights**, with alpha=1, 5 and 20. It
averages absolute log errors over six banks and the three conditions
(m,n)=(1024,2048), (1024,4096), (2048,4096), using each alpha's arithmetic
mean disagreement over 20 A/B pairs. Its 18 values are independently rebuilt
from scalar full/half reference forecasts, source-count split weights, and
pair scores. `ALPHA_SWEEP.md` explains the full protocol and arrays.

The synthetic sensitivity controls in E.2 recompute smoothing-dependent
weights from the same auxiliary samples. This differs from Table 3's fixed
weights. Their population moments isolate approximation errors within the
specified synthetic generator. They do not identify the cause of error in
a particular real-text corpus.

## Scope

The numerical records needed for the offline rebuild are included. Real-text
forecast generation, new neural inference and training require external text
and other inputs documented in `DATA_PREPARATION.md`. No corpus text,
decodable token arrays, target-token ranks, neural weights or manuscript are
included. Alpha records beyond the values selected for Table 3 remain
available so reviewers can inspect the full grid without reading Appendix F.

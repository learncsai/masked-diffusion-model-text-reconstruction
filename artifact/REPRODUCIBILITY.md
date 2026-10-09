# What the reproduction command checks

`python reproduce.py` verifies every delivered file against its SHA-256,
creates a working copy, recomputes the following results and reconciles table
rows with the separately submitted manuscript. No external dataset is read.
The inventory covers the main text and Appendices A through E: eight empirical
tables, two definition tables and five figures. Appendix F is omitted.
Full alpha records remain available as supporting evidence.

| Component | Recomputed evidence |
|---|---|
| 01 Occupancy | Query-count histograms, including E.3's 7.3% unseen and 20.8% below-five-count WikiText statistics. The former six-row table is now a diagnostic |
| 05 Top-k | All 63 corpus/size/k conditions from per-chunk hit rates, with equal weights for chunks and reconstructors |
| 06 MDLM | MSE, disagreement, input rankings, shared seed/initialization identities, quality tables and plots from per-chunk/model records |
| 07 Directions | Rank-4, dimension-64 test at n=2048 and 12000 updates. Eigenvectors, source-fold capture and independent SVD checks from exact covariance blocks |
| 09 Correction | Bank/pair calibration, intervals and size choices. Synthetic decomposition checks use saved aggregate components |
| 10 Follow-up | Bootstrap batches, direct split estimates, fixed inflation, correction policies, paired intervals, size decisions and fresh confirmation |
| 11 Oracle/multiplier | Exact synthetic population moments, enumeration checks, oracle errors, multiplier sweeps and transfer intervals |
| 12 Bootstrap confirmation | Batch corrections, source-count weights, confirmation comparisons, Monte Carlo summaries and retrospective original-grid switch |
| 13 Alpha sweep | Main Table 3 independently reconstructed from scalar full/half-bank forecasts and 20 pair scores. E.3's paired slope and ranking checks. Full-grid summaries and extra plots are diagnostic outputs. ALPHA_SWEEP.md explains the records and scope. |

`APPENDIX_E_CHECKS.json` records the independent Table 3 calculation and retained
E.1/E.3 numerical claims. `reproduced/INDEX.md` maps outputs to submission numbers.

The script fails if an empirical table is missing, a generated row differs,
a required row is omitted or a figure has no generator. It checks neural size
response and directional percentages in the prose and sums all training logs.
The image/text correspondence and data-role tables are definitions, so the
script copies them without treating them as experiments.

## Exact reductions

All natural-language token arrays, individual target ranks and query order
were removed. Top-k retains per-chunk hit rates. Occupancy retains unordered
histograms of queried row totals. Directions retain only the two 64 by 64
source-fold covariance blocks needed for the reported test. These transformations
preserve the reported calculations. Original hashes and transformations are
recorded in REDUCTIONS.json.

Real-text records are measurements of earlier forecasts or models, not inputs
for new forecasts. Synthetic oracle moments are recomputed from the population
model and seeds. The missing-context decomposition uses saved component
summaries rather than rerunning its simulation.

## Sampling and uncertainty

Sources stay together when the original forecasts resample data. A source is
a story, a WikiText paragraph or a news article, and can supply several chunks.
Only the final source may be truncated at the target budget. Jackknife halves
are disjoint sets of sources and use source-count weights.

Correction-study intervals resample independent banks and independent A/B
pairs within each corpus, keeping methods and nested sizes together. Auxiliary
data, evaluation inputs and saved Monte Carlo draws are fixed. Monte Carlo
standard errors are reported separately. Evaluation chunks, nested sizes and
draws do not add independent corpus pairs.

Available identifiers and hashes support allocation checks. Historical overlap
audit reports are supplied where complete inputs are omitted. The default
script does not repeat those omitted-input audits. Statistical independence
of real sources remains a modeling assumption beyond exact overlap.

## Earlier audit and full replication

`docs/historical_audits/` records the formula/code review, 46 focused tests,
27 training-condition checks and CPU rescoring of six checkpoints. Those used
local inputs and weights. They are records of that audit, not tests silently
rerun by this compact release.

DATA_PREPARATION.md gives acquisition, canonicalization, chunking and hash
verification. TRAINING.md gives the neural runtime. Complete input rosters,
token arrays and weights are not delivered. Fresh downloads must match
recorded hashes before claiming exact reconstruction. The optional recovery
helper restores one retained allocation and fails on a mismatch.

Comparisons keep their original timing designations. Rebuilding results does
not make a comparison on known outcomes prospective. Source and multinomial
formulas forecast the lag reconstructor. Their MDLM comparison is an empirical
transfer test.

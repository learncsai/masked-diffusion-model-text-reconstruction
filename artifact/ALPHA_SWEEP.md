# Table 3 and Appendix E.3: sensitivity to smoothing

Component `13_alpha_sweep` contains the numerical records, configuration,
protocol and code for the real-text alpha sweep. The root `python reproduce.py`
command rebuilds main Table 3 and checks the alpha findings retained in E.3.
The full sweep's three additional tables and five figures are also generated
on CPU under `reproduced/diagnostics/`. Appendix F is not submitted.

## Reproduce Table 3

Open `reproduced/tables/table_3.csv` or
`reproduced/tables/tab_main-alpha-ablation.tex`. The independent calculation
in root `rebuild_appendix_e.py` reads the full/half reference forecasts and
source-count weights in `analytic_bank*_m*.npz`. It recomputes the jackknife
as twice the full estimate minus the source-weighted half estimates.
Observed disagreement is the arithmetic mean of `disagreement` over the
20 `observed_pair*.npz` files.

For each corpus and alpha=1,5,20, select the **fixed** weight arm, corrected
source and corrected multinomial, and the three cells
(m,n)=(1024,2048), (1024,4096), (2048,4096). Average the absolute log
forecast/observation ratio over six banks and those three cells. Apply the
absolute value before averaging. Round only the final mean to three decimals.
This produces all 18 table values. The script reconciles them against the
full-sweep `calibration_by_regime.csv` and the manuscript fixture.

`reproduced/diagnostics/table_3_bank_cells.csv` contains all 324 contributing
bank/cell losses. `reproduced/APPENDIX_E_CHECKS.json` records full-precision
values, paired intervals and the checks of E.3's other retained findings.

## Design

The sweep uses the original 20 A/B pairs and six independent reference banks
per corpus. Corpus sizes are 512, 1024, 2048 and 4096 chunks, and reference
sizes are 1024, 2048, 4096 and 8192 chunks. Chunks contain 64 tokens. Sizes are
nested within each pair or bank. The shared evaluation set has 128 chunks and
mask rate 0.5. The lag radius is two.

The smoothing values are 0.5, 1, 2, 5, 10, 20 and 50. The primary arm keeps
the alpha=5 reconstruction weights fixed. The robustness arm recomputes the
weights from the same auxiliary data at each alpha, using the original
trace-scaled ridge rule. The unigram, masks and all corpus allocations stay
fixed. This step trains no neural model.

Both analytic laws and their source-split corrections cover the full grid.
Nonlinear source bootstrap and its correction use alpha=1, 5 and 20, with
three independent batches of 16 draws. Within a batch the same whole-source
draws are used across smoothing values and weight arms. The reported estimate
averages the three batch variance estimates. Its Monte Carlo standard error
is reported separately.

All new-alpha forecasts and the analysis code were saved before scoring the
new-alpha A/B outcomes. Alpha=5 outcomes and these allocations had already
been studied. This is a sensitivity experiment on existing allocations,
not an independent confirmation on newly collected corpora.

## Rebuilt outputs

Inside `reproduced/work_<timestamp>/13_alpha_sweep/results/alpha_sweep_v1/rebuilt/`:

| File | Contents |
|---|---|
| slopes.csv | Observed and forecast log-log size slopes for every alpha and both weight arms |
| hypotheses.csv | Paired slope-flattening and slope-accuracy contrasts |
| quality.csv | MSE, relative MSE improvement, exact top-1 accuracy and weight norms |
| calibration.csv | Forecast/observation ratios and errors for every bank-averaged condition |
| calibration_by_regime.csv | Errors on the full grid and separately for m<n, m=n and m>n |
| paired_calibration.csv | Paired error differences relative to plain multinomial |
| row_bins.csv | Same-lag energy and size elasticity in auxiliary-defined frequency bins |
| coverage.csv | Both-seen and one-sided same-lag energy, with signed cross-lag energy separate |
| monte_carlo.csv | Bootstrap batch estimates and Monte Carlo standard errors |
| reference_diagnostics.csv | Query-weighted reference counts relative to alpha |
| tables.json | Rows for three extended diagnostic tables, outside the submission inventory |
| validation.json | Dimensions, chronology checks and energy partition identities |

The `reproduced/diagnostics/alpha_sweep/` folder contains five PDFs whose names begin
`fig_alpha_`. The full grid is retained even where the paper shows selected
alpha values. Nonpositive corrected forecasts are counted as failures and
excluded from logarithms. They are not clipped to a positive constant.

## Uncertainty and interpretation

Intervals use 5000 crossed resamples of entire A/B pairs and entire reference
banks within each corpus. Each draw keeps all methods, smoothing values,
weight arms and nested sizes together. They condition on the shared auxiliary
and evaluation data. They do not treat chunks, smoothing settings or bootstrap
draws as additional independent corpus pairs.

The size slope regresses log arithmetic mean disagreement on log corpus size.
The primary forecast uses m=8192 without correction. Corrected m=2048 slopes
are secondary. Calibration error averages the absolute log ratio of each
bank forecast to the mean disagreement over 20 pairs. Regime averages weight
grid cells equally. A smaller slope error or calibration error is better.

The row-bin Poisson curves are a conditional-multinomial reference model.
They use auxiliary row frequencies and conditional probabilities. They are
not an independence assumption imposed on the source-level forecast. A bin's
coverage share divides by same-lag energy. Signed cross-lag terms are reported
separately because they are not nonnegative variance components.

## Local validation and full rerun

`local_validation.json` records the comparison against the original alpha=5
analytic forecasts, source-bootstrap batches, disagreements and MSE values.
`method_saved.json`, `analysis_saved.json` and `forecasts_saved.json` record
code hashes and ordering. They are local audit records, not external
preregistration certificates.

`presentation_amendment.json` records a later tick-label formatting change
made after inspecting the rendered figures. It changes no measurements or
statistical calculations. The original plotting and validation files remain
under `analysis_code/`, and the numerical replay checks both versions' hashes.
`summary/row_interval_audit.json` records that all 5000 row-bin resamples
were defined for every reported positive-energy contrast.

The compact component supplies scalar forecasts, pair scores and numerical
diagnostics. It omits the original text and token arrays. With separately
reconstructed original inputs and the original alpha=5 audit arrays, the
scientific runner is:

```sh
python scripts/run_alpha_sweep.py prepare --corpus tinystories
python scripts/run_alpha_sweep.py forecast --corpus tinystories
```

Repeat both stages for `wikitext` and `cnn_dailymail`, then save the analysis
code hashes, run `python scripts/run_alpha_sweep.py freeze`, and score each
corpus with `python scripts/run_alpha_sweep.py score --corpus CORPUS`.
`DATA_PREPARATION.md` describes the upstream allocations and hash requirements.
Use a new output directory for a new experiment. The delivered numerical
replay needs none of those external inputs.

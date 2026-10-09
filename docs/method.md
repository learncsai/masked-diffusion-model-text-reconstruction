# Method and metrics

This guide follows Sections 2-5 and Appendices A-E of the submitted paper. [The paper map](paper-map.md) connects these concepts to outputs.

## Local reconstruction

The vocabulary is GPT-2's 50,257 content IDs, including EOS. Inputs are 64-token chunks. A hidden token is predicted from visible neighbors at offsets `-2, -1, 1, 2`. Counts are collected only within a chunk; source grouping joins the chunks contributed by the same story, paragraph, or article.

For a context token `b`, target token `a`, and offset `delta`, the smoothed row is:

```text
P_delta(a | b) = (N_delta(b, a) + alpha * p[a]) / (N_delta(b, +) + alpha)
f_i = p + sum_delta w_delta * (P_delta(. | visible_neighbor) - p)
```

The default `alpha=5` pulls rare rows toward a shared unigram `p`; unseen contexts return `p`. Auxiliary sources estimate `p` and fit weights using ridge `0.01`. The same quantities are fixed for A and B. Count outputs are affine scores and may be negative. MSE and top-k accuracy remain meaningful; these scores are not passed off as probabilities or scored with log loss.

Implementation: `src/diffusion_lm_rmt/sparse_categorical_lag.py` and `categorical_lag.py`. Sparse residuals plus a multiple of `p` give exact full-vocabulary norms without allocating every dense output vector (Appendix E.4).

## What disagreement measures

For each evaluation chunk, average the squared Euclidean distance between A and B output vectors over its hidden positions, then average equally over chunks. The expectation over independently collected A/B corpora is the forecast target. Size `n` is the number of chunks **in each arm**, and `m` is reference-bank chunks. Neither counts independent sources.

Reconstruction MSE uses squared distance to the one-hot target. Relative improvement is `1 - MSE(model)/MSE(unigram)`. The shared unigram has zero A/B disagreement and zero relative improvement. Top-k accuracy measures whether the true token is among the k largest output scores. Quality and stability describe different properties.

## Forecasts and correction

The source law linearizes smoothed rows while retaining random row totals, source length, within-source dependence, and cross-lag covariance (Section 3, Appendix A). A same-lag ablation removes cross-lag terms. The multinomial law approximates count variability at the row level. The conditional row collision is evaluated over all output classes and held fixed as target size changes.

An independent reference bank estimates the quantities in these laws. The source-split jackknife partitions complete sources into two groups and combines estimates as twice the full-bank estimate minus the source-count-weighted half-bank estimates. It addresses first-order finite-reference bias; it cannot recover a context unseen in the whole bank.

The nonlinear source bootstrap instead resamples complete sources, refits all lag rows jointly, and estimates output variance from 48 replicates in three batches of 16. Only the final source is truncated to meet the chunk budget. Its jackknife uses the same source partition as the analytic corrections.

Implementation: `source_moment_forecast.py`, `reference_correction.py`, `alpha_sweep.py`, and the correction runners in `scripts/`.

## Analysis units

Calibration error is the absolute log of forecast divided by mean A/B disagreement. Reference banks, grid cells, and corpora receive the documented equal weights. Crossed bootstrap intervals use 5,000 draws: resample whole banks and whole A/B pairs independently while retaining method pairing and nested sizes. Intervals condition on auxiliary data, evaluation inputs, masks, and saved Monte Carlo draws. Evaluation chunks from a common source are not treated as independent corpus replications.

## Neural comparison

MDLM uses the full visible chunk and returns probabilities. A/B models share auxiliary initialization, minibatch-index streams, diffusion-time streams, masks, optimizer settings, and update counts. Their corpus contents differ. Three pairs per corpus and size allow matched disagreement, reconstruction quality, input ranking, and directional overlap comparisons (Section 5, Appendices C-D). See [training](training.md) for the exact architecture and schedule.

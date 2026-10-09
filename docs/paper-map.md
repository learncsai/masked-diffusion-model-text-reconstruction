# Map to the submitted paper

References here follow the supplied **finalaistats_overleaf__2_.pdf**, not the older archive inventory. The PDF and manuscript source are excluded from Git. The reference SHA-256 is recorded in [repository provenance](../provenance/repository_origin.json).

## Tables

| Submitted table | Subject | Paper location | Compact evidence | Generator |
|---|---|---|---|---|
| 1 | Source-disjoint data roles | Section 2 | Protocol definitions | Root `reproduce.py` exports definitions |
| 2 | Calibration of ten forecasts | Section 4.2 | `artifact/components/10_correction_followup/` | `artifact/rebuild_followup.py` |
| 3 | Smoothing ablation, alpha 1/5/20 | Section 4.2; E.3 | `artifact/components/13_alpha_sweep/` | `artifact/rebuild_appendix_e.py` independently recalculates all 18 entries |
| 4 | Matched count/MDLM disagreement and MSE improvement | Section 5 | `artifact/components/06_mdlm/` | Component `scripts/build_mdlm_paper_results.py` |
| 5 | Calibration intervals and conditional corrections | B.2 | Components 10 and 12 | `artifact/rebuild_followup.py` |
| 6 | Fresh-source confirmation and added comparators | B.3 | Components 10, 11, 12 | `artifact/rebuild_bootstrap_confirmation.py` |
| 7 | Correspondence to image diffusion | C.3 | Concept definitions | Root `reproduce.py` exports definitions |
| 8 | MSE and top-k quality | D.1-D.2 | Component 06 and matched top-k records | Component `scripts/build_mdlm_paper_results.py` |
| 9 | Exact-population forecast calibration | E.2 | `artifact/components/11_oracle_multiplier/` | `artifact/rebuild_oracle_multiplier.py` |

The archive's size-decision table has no table number in this submitted version. Its calculation is retained under `results/reproduction/diagnostics/`, along with extra smoothing settings and earlier studies.

## Figures

| Submitted figure | Subject | Paper location | Rebuilt filename | Evidence |
|---|---|---|---|---|
| 1 | Analytic calibration at n=4096 | Section 4 | `figure_1.png` | Component 09 |
| 2 | Reference-by-target calibration grid | B.1 | `figure_2.png` | Component 09 |
| 3 | Multiplier transfer | B.3 | `figure_3.png` | Component 11 |
| 4 | Matched count and MDLM size response | C.1 | `figure_4.png` | Component 06 |
| 5 | Top-k reconstruction accuracy | D.2 | `figure_5.png` | Component 05 |

Figure 4 in the submitted paper is called Figure 2 in the archive's original guide. The root runner corrects this order and the analogous table-number shifts.

## Experiment coverage by section

| Section or appendix | Implementation and evidence |
|---|---|
| Section 2, Appendix A | Sparse reconstructor, count influences, source moments, exact identities; numerical tests |
| Section 3, B.1-B.2 | Source/multinomial laws, source-split corrections, nonlinear bootstrap, direct splits, inflation baselines |
| Section 4, B.3 | Twenty-pair/six-bank study; fresh six-pair/three-bank confirmation; multiplier transfer and size choices |
| Section 5, C.1 | Recorded three-corpus MDLM implementation; 66 logs; disagreement, quality, and input rankings |
| C.2 | Two-fold rank-4 overlap in dimension 64, with unigram/frequency control; component 07 |
| C.3 | Conceptual image/text mapping; no new image-diffusion experiment is claimed |
| B.4-B.5 | Source allocation; earlier mask, size, and reference-ratio studies; `artifact/earlier_results/` |
| D.1-D.2 | Full-vocabulary MSE and exact per-chunk top-k measurements |
| E.1-E.2 | Controlled populations, missing-context decomposition, exact population oracles |
| E.3 | Full seven-alpha sweep, fixed/refit weights, row coverage, paired slope comparisons |
| E.4 | Sparse identities, implementation audits, CPU/GPU checks, training provenance, reproduction limits |

There is no Appendix F in the submitted inventory. Extended sweep records remain available as supporting material. Original archive guides are preserved for provenance and may use older numbering; this file defines repository-facing references.

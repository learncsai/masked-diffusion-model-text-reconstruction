# Study allocations and reuse

This guide preserves the allocation overview moved out of the paper.
It describes historical experiments, not additional independent observations.
The compact release retains numerical summaries and the core study records.
Earlier full allocation arrays are not all included.

## Study overview

Counts are per corpus. Real-text sizes count 64-token chunks per A/B arm or reference bank.

| Study | Pairs and target sizes | References | Shared quantities and timing |
|---|---|---|---|
| Correction study | 20 pairs, n=512/1024/2048/4096 | Six banks, m=1024/2048/4096, plus 8192 baseline | Matched auxiliary split. Source, source jackknife, and multinomial forecasts precede scores. Other baselines reuse these outcomes. |
| Fresh confirmation | Six new pairs, n=1024/2048 | Three new banks, m=1024/2048 | Same matched auxiliary/evaluation task. Corrected multinomial selected as primary before scoring. |
| Original calibration and quality | Two pairs across three sizes/mask rates, plus seven-pair n=512 mask test | m=2048 | Original auxiliary role with study-specific partitions/masks. No prospective claim. |
| Fresh size response | Seven pairs at n=512/2048, then 7/5/4 at 4096 | m=2048/6144 | Saved original-grid context. Forecasts precede scoring. Extensions reuse pairs. |
| Matched counts, neural transfer, top-k | Three pairs, n=512/1024/2048. Count extension to 4096/8192 | m=2048 to 8192 | Matched auxiliary split. First three fresh-size pairs. Reference enlargement and neural comparisons follow earlier scores. |
| Reference-ratio follow-up | Three known pairs and three fresh pairs | Nested references up to 12288 and separate 4096 bank | Matched split. Includes both known-outcome cells and forecasts preceding fresh scores. |
| Multiplier transfer | Original and fresh correction-study pairs, no new scoring | Same saved banks | Retrospective. Factors selected on named original training cells/corpora, then applied to held-out measurements |
| Exact population oracles | Same 80-pair controls at n=256/1024, five cases | Exact population quantities replace reference estimates | Retrospective. Same population, auxiliary data, masks and A/B outcomes as existing controls |
| Synthetic diagnostics | 80 pairs/four banks for correction development. 200 pairs/eight banks for propagation | Study-specific reference grid | K=512, L=8. Independent synthetic auxiliary data. The designs are distinct. |

## Shared definitions

- A pair contains one A corpus and one B corpus. The size `n` is the number
  of chunks on **each** side. Reference size `m` is also a chunk count.
- Real-text chunks have 64 tokens. The three corpora are TinyStories,
  WikiText-103 and CNN/DailyMail. Sources are stories, paragraphs and articles.
- O denotes the original auxiliary data role, identified as `replicate3_train_a`
  in the original count configurations. Its sources are divided between row
  estimation and weight validation using study-specific seeds. O does not
  imply a common partition or common masks across the original studies.
  In particular, the original grid and initial seven-pair mask test have
  different auxiliary-partition and mask hashes in their `pair_results.csv`.
  The prospective size studies keep the grid's saved 512-chunk, mask-rate-0.5
  context from the calibration diagnostics.
- M denotes the later matched auxiliary partition in the per-corpus input
  manifests. It is shared by the matched count, neural and reference-ratio
  studies. The neural parent is trained on its auxiliary training portion.
  O and M reuse the same underlying auxiliary role but have different
  train/validation partitions. They do not give independent auxiliary samples.
- S denotes the synthetic control's independent auxiliary samples: 8,192
  chunks for the unigram and two 2,048-chunk samples for the weights.
- Real-text studies share the same 128 evaluation chunks. The original grid
  and initial seven-pair test use mask rates 0.2, 0.5 and 0.8. Later reported
  studies use the saved mask at 0.5. The synthetic control uses 64 chunks.

## Locate the evidence

- Current correction, follow-up, oracle and bootstrap confirmation records:
  components 09 through 12, with configurations and protocols.
- Matched neural inputs, identity records and per-chunk measurements:
  component 06. Top-k hit rates: component 05. Reported direction blocks:
  component 07.
- Earlier calibration, fresh-size, large-size and reference-size summaries:
  earlier_results/, grouped by the original component name.
- Earlier reference-ratio cells and their stage labels:
  earlier_results/08_reference_ratio/results/reference_ratio_v1/.
- Preparation and statistical implementations: research_code/. The exact
  archived neural training implementation: training_code/.
- Historical implementation checks: docs/historical_audits/.

The compact replay does not repeat every earlier source-data audit.
DATA_PREPARATION.md states which inputs are required for a full replication.

## Interpretation of prospective status

“Prospective” describes forecasts saved before scoring the corresponding A/B
outcomes. None of these studies is externally preregistered. The main text
distinguishes an initial exploratory observation from the subsequently
specified reference-ratio follow-up. Earlier prospective outcomes became
discovery data for that later hypothesis.

Extending a previously scored pair is not a new independent pair. The
reference-sensitivity study changes only the reference on fixed large-size
outcomes. The new reference-ratio study contains both known-outcome forecast
cells and genuinely fresh A/B allocations, and reports both reference banks.
No claim of prospective chronology is made for the original grid, the initial
mask test, neural transfer or the simulation diagnostic.

Coverage, timing, shuffled-order results, input rankings, top-k scores and
directional energy do not add independent A/B pairs. The corresponding rows
and subsection descriptions identify which allocations they reuse.

## Finite-reference correction study

Component 09 uses 20 new A/B pairs and six source-separated reference banks
per corpus. Target sizes are 512, 1024, 2048, and 4096 chunks. Reference sizes
are 1024, 2048, and 4096, with an uncorrected 8192-chunk reference baseline.
Nested sizes retain their shared observations in inference. The auxiliary
statistics and 128 masked evaluation chunks match the earlier matched task.

The primary method and every forecast and stability target were saved before
new A/B outcomes. The source jackknife, full source, and uncorrected multinomial
form the primary comparison. Multinomial jackknife was added afterward, using
the identical source partition and coefficients without tuning. The package
keeps this follow-up explicitly exploratory.

The synthetic correction controls use 80 pairs, four independent banks per
reference size, K=512, and L=8. Their source allocations differ from the older
200-pair known-population control. Complete frozen configurations and the
prospective protocol are included in component 09.

## Added baselines and fresh confirmation

Component 10's `configs/correction_followup_v1.json` and
`docs/correction_followup_protocol.txt` specify the follow-up. The new
bootstrap, direct-split, fixed-inflation, correction-policy and absolute-target
analyses reuse all twenty-pair outcomes in component 09. They are exploratory.
All twelve reference/target cells per corpus are reported in `calibration.csv`.

The separate confirmation allocates six new A/B pairs and three new banks per
corpus, with 1024/2048 nested chunks in each. All new roles are separate from
every earlier role by source, text and chunk hashes. `validation.json` records
that local audit. The source totals are 10638, 13556 and 15376 in corpus order.
Its primary condition is m=1024, n=2048 and its primary method is corrected
multinomial compared with plain multinomial. The method, all forecasts and
analysis code were saved before its outcomes. The original exploratory
multinomial comparison is not retrospectively reclassified.

## Bootstrap extension on the separate confirmation pairs

Component 12 reuses component 10's three banks and six pairs per corpus at
m=2048,n=1024. New nonlinear source-bootstrap forecasts and their source-split
corrections use the original resampling procedure, source partitions and draw
count. Settings and code hashes were saved before the new computation, after
the outcomes were known. This added comparison is not a prospective trial.
All nine banks and all three corpora are reported, including fixed/calibrated
multinomial comparators. Its inference and the original-grid switching rule
are documented in BOOTSTRAP_CONFIRMATION.txt. The rule is retrospective and
only its larger-reference branch is checked on the confirmation allocation.

## Smoothing sensitivity: main Table 3 and Appendix E.3

The submission retains the compact ablation and its protocol in E.3. The
full sweep is numerical supporting material. The extended Appendix F is not
submitted. ALPHA_SWEEP.md and APPENDIX_E.md map the retained findings.

Component 13 reuses the 20 pairs and six reference banks from component 09.
It evaluates alpha=0.5,1,2,5,10,20,50 on four target sizes and four reference
sizes, with fixed and auxiliary-refit weights. Bootstrap uses alpha=1,5,20.
All new-alpha forecasts and analysis code precede new-alpha A/B scoring.
The existing alpha=5 outcomes remain known, and no fresh allocation is claimed.
ALPHA_SWEEP.md explains the complete numerical records, paired inference,
row-frequency diagnostics and exact reproduction of alpha=5 measurements.

# Data preparation and full experiment replication

The default `python reproduce.py` needs no source corpora: compact measurements
suffice for all paper tables and figures. This guide documents how experiment
inputs were constructed and what is needed to repeat earlier stages.

## 1. Obtain upstream sources

| Corpus | Hugging Face dataset | Configuration | Split/field | Sampling unit |
|---|---|---|---|---|
| TinyStories | roneneldan/TinyStories | Default | train/text | Story |
| WikiText-103 | Salesforce/wikitext | wikitext-103-raw-v1 | train/text | Paragraph |
| CNN/DailyMail | abisee/cnn_dailymail | 3.0.0 | train/article | Article |

Use GPT-2's tokenizer (openai-community/gpt2). DATA_SOURCES.md and
LICENSE_REVIEW.txt give attribution and published terms. No dataset text or
tokenizer vocabulary is distributed here.

Choose and record immutable dataset and tokenizer commits when acquiring data.
Historical downloads did not pin every revision, so a new download is not
assumed to match. Verify retained source and chunk hashes. A mismatch must be
reported as changed input in a new replication.

## 2. Canonicalize and identify sources

Read the unshuffled upstream train stream in order:

```python
canonical = " ".join(text.replace("\r", "\n").split())
digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
source_id = f"{corpus}:{upstream_row_index}:{digest[:16]}"
```

The index belongs to the **unfiltered** stream. Deduplicate complete sources
by the canonical-text hash. TinyStories accepts nonempty text. WikiText and
CNN/DailyMail require at least 400 canonical characters. WikiText excludes
headings whose canonical text starts with "=". The study's WikiText sampling
unit is a paragraph, not a complete Wikipedia article.

Implementation: research_code/src/diffusion_lm_rmt/neural_data.py and its
archived training_code/src version.

## 3. Tokenize, chunk and keep identities

Encode without automatic special tokens, append GPT-2 EOS (50256), split each
source into non-overlapping **64-token** chunks and discard the incomplete
tail. Do not cross source boundaries. Output vocabulary size is 50257.
Input-only MASK/PAD additions have IDs 50257/50258 and are excluded from output
classes. The vocabulary hash is in docs/tokenizer_metadata.json.

For each retained chunk store document_id, its original chunk_index within
the source, and SHA-256 of little-endian int32 token bytes. Exact duplicate
chunks are excluded within a split and against previously reserved roles.

CNN/DailyMail caps an article at two spaced chunks. If a source has C chunks
and a cap c<C, select indices
`min(C-1, int((j+0.5)*C/c))` for j=0,...,c-1 before duplicate checks.
This samples across the article body. Preserve recorded indices when recovering
an old allocation instead of taking the first two chunks.

## 4. Assign roles by source, then apply chunk budgets

| Role | Purpose | Shared between A/B? |
|---|---|---|
| Auxiliary | Shared unigram p, row/weight training and validation, neural parent | Yes |
| Reference | Estimate sampling variation for forecasts | Yes |
| A and B | Separate count tables or neural models | Independent source allocations |
| Evaluation | Common masked inputs and targets | Yes |

Assign each source to one role and check source IDs, text hashes and chunk
hashes across roles. Draw independent A/B pairs and reference banks using the
study configuration, excluding previously reserved sources. Nested sizes use
prefixes of the same ordered allocation. A source can supply multiple chunks
but cannot cross roles. Only the final source may be truncated to hit a chunk
budget. Population source contributions are defined before this truncation.

Size n counts chunks **in each A/B arm**, and m counts reference chunks.
These are not independent-source counts. Default lag radius 2 uses offsets
-2,-1,1,2 and is separate from chunk length.

Matched studies share auxiliary quantities, 128 evaluation chunks and the
saved q=0.5 mask. The original grid also uses q=0.2 and 0.8. Smoothing alpha=5
and weight ridge=0.01. Original and matched studies have different auxiliary
partitions. STUDIES.md explains their reuse.

## 5. Use study-specific settings

| Study | Independent allocations per corpus | Chunk sizes |
|---|---|---|
| Matched count/MDLM/top-k | Three A/B pairs | n=512,1024,2048; count extensions to 4096,8192 |
| Correction study | 20 pairs, six reference banks | n=512,1024,2048,4096; m=1024,2048,4096 and 8192 baseline |
| Fresh confirmation | Six new pairs, three new banks | n=1024,2048; m=1024,2048 |
| Added bootstrap confirmation | Same confirmation allocations | m=2048, n=1024 |

research_code/configs and research_code/docs contain settings and protocols.
The correction study uses seed 20261003 and a 128000-source preparation pool.
Fresh confirmation uses seed 20261017 and excludes earlier allocations.
Changing the pair count can change the permutation and downstream roles.
Use retained ordered rosters for historical reconstruction.

Bootstrap whole source groups with replacement until n chunks are collected,
truncating only the last source. Rebuild all lag tables jointly to preserve
overlap and cross-lag dependence. Jackknife halves split **sources** and use
source-count weights. Do not resample individual tokens or treat chunks from
one source as independent draws.

MDLM A/B models share parent initialization, optimizer settings, update counts
and random streams for minibatch indices, masks and time sampling. The corpus
contents differ. TRAINING.md gives the archived code and RTX 4090 runtime.

## 6. Recover one retained allocation

recover_corpus_chunks.py is optional and never called by reproduce.py. It
requires datasets and transformers in addition to the analysis requirements:

```sh
python recover_corpus_chunks.py --corpus cnn_dailymail --revision DATASET_COMMIT --tokenizer-revision TOKENIZER_COMMIT --records PATH_TO_REFERENCE_ROSTER.json --output local_data/reference.npz
```

Reference rosters are under components/12_bootstrap_confirmation/data/diffusion_llm/correction_followup_v1/.
Evaluation records are in component 06. Choose an ordered JSON list with
document_id, chunk_index and sha256 entries, and replace the placeholders
with that path and the acquired commits. The helper restores original order,
checks every source ID and chunk hash, writes a verification report and fails
on a mismatch. It never overwrites an existing output. Keep recovered files
outside this ZIP.

This restores one allocation, not the entire experiment. Complete historical
A/B rosters and all auxiliary masks for the correction studies are omitted.
Manifests identify missing inputs but cannot replace them. Exact historical
reconstruction needs those allocations. A new replication can instead follow
the documented protocol and record new allocations. New neural inference
requires weights, and full training requires the inputs in TRAINING.md.

## 7. Verify a full rerun

Check vocabulary, source/chunk identities, role separation, masks and auxiliary
quantities before generating forecasts. Save configurations and forecast hashes
before scoring new outcomes. Compare generated measurements with components/
records, retaining method pairing, nested sizes and reference-bank identities
in inference. Report changed hashes, software or GPU behavior explicitly.

No download, new real-text forecast, neural inference or training is performed
by the default numerical reproduction command.

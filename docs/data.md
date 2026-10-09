# Data sources and preprocessing

## Acquire source data separately

| Corpus | Upstream dataset | Configuration | Split / field | Independent sampling unit |
|---|---|---|---|---|
| TinyStories | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Default | `train` / `text` | Story |
| WikiText-103 | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) | `wikitext-103-raw-v1` | `train` / `text` | Retained paragraph |
| CNN/DailyMail | [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) | `3.0.0` | `train` / `article` | Article |
| Tokenizer | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | GPT-2 | Tokenizer files | 50,257 output classes |

TinyStories is synthetic text generated using GPT-3.5/GPT-4. WikiText's paragraph grouping does not capture dependence across a whole Wikipedia article. CNN/DailyMail uses the cased, non-anonymized article configuration, with at most two spaced chunks per article. The study does not train on article summaries.

Upstream cards and access links were checked when assembling this repository. For attribution, recorded card revisions, and the distinctions between software/dataset metadata and underlying article terms, see [DATA_SOURCES.md](../artifact/DATA_SOURCES.md), [LICENSE_REVIEW.txt](../artifact/LICENSE_REVIEW.txt), and [NEWS_DATA_ACCESS.md](../artifact/NEWS_DATA_ACCESS.md). The submitted paper records differences between WikiText's metadata and its licensing paragraph. This repository distributes measurements and source hashes rather than corpus text or reversible token sequences.

## Canonicalization and filtering

Read the upstream training stream in its original order. Preserve the **unfiltered** row index in source IDs:

```python
canonical = " ".join(text.replace("\r", "\n").split())
digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
source_id = f"{corpus}:{upstream_row_index}:{digest[:16]}"
```

Deduplicate whole sources by canonical-text hash. Keep nonempty TinyStories records. Require at least 400 canonical characters for WikiText and CNN/DailyMail. Remove WikiText headings starting with `=`. Hashes identify a particular upstream text and ordering; a different dataset revision may change them.

## Tokenization and chunking

Encode with GPT-2 without automatic special tokens, append EOS `50256`, split into non-overlapping **64-token chunks**, and discard an incomplete tail. Never cross source boundaries. Add input-only MASK `50257` and PAD `50258`; both are excluded from the output softmax and reconstruction vocabulary.

For each retained chunk, record `document_id`, the original `chunk_index`, and SHA-256 of little-endian int32 token bytes. Exclude exact duplicate chunks within a role and across reserved roles. Preserve original chunk indices after filtering.

For a news article with `C` full chunks and a cap `c=2`, use spaced indices `min(C-1, int((j+0.5)*C/c))` for `j=0,1` when `C>2`. A source is consumed by one role even if its last chunk is truncated by the role's budget.

## Source-disjoint roles

Assign sources before applying chunk budgets. Auxiliary data establishes the shared unigram and weights; an independent reference estimates variation; A/B arms provide independent count tables or training corpora; evaluation supplies identical masked inputs. Reference banks and all A/B arms also remain mutually source-disjoint. Nested sizes use prefixes within one role, rather than independently allocating every size.

The matched task has 128 evaluation chunks and one saved Bernoulli mask per chunk at `q=0.5`; force position zero hidden if a mask is empty. Earlier studies also examine `q=0.2,0.8`. Radius `R=2`, smoothing `alpha=5`, and weight ridge `0.01` are fixed defaults. Auxiliary sources are split by source into row/unigram estimation and weight fitting.

## Prepare a new replication

After installing the [experiment environment](setup.md), run from a fresh clone:

```bash
python scripts/prepare_public_data.py --corpus tinystories
python scripts/prepare_public_data.py --corpus wikitext
python scripts/prepare_public_data.py --corpus cnn_dailymail
```

The command resolves and records immutable dataset/tokenizer commits before reading data. For an explicitly chosen revision, add `--dataset-revision COMMIT --tokenizer-revision COMMIT`. Use the same tokenizer commit across all corpora. Resolved commits, preparation hashes, and audits are written under ignored `data/acquisition/`.

This creates a **new replication**, using a 64,000-source matched preparation block and a separate 128,000-source correction block. It prepares three matched A/B pairs up to 8,192 chunks, a 2,048-chunk matched reference, shared auxiliary/evaluation inputs, and the twenty-pair/six-bank correction inputs. It saves source rosters and excludes duplicate source texts and chunks across roles. The archived confirmation runner can allocate unused sources from the correction block after these inputs exist.

The new entry point uses the paper's source/chunk rules and archived scientific settings. It does not recreate the historical permutation history or every earlier-study auxiliary partition. Acquisition and tokenization can take substantial CPU time and several GB of disk space. Data is local and ignored by Git. A failed or changed preparation requires a fresh clone/worktree; the script refuses to overwrite existing input roles.

## Recover a recorded allocation

For a roster retained in the compact archive, use its recovery helper:

```bash
python artifact/recover_corpus_chunks.py --corpus cnn_dailymail \
  --revision DATASET_COMMIT --tokenizer-revision TOKENIZER_COMMIT \
  --records artifact/components/12_bootstrap_confirmation/data/diffusion_llm/correction_followup_v1/cnn_dailymail/reference0.json \
  --output local_data/cnn_reference0.npz
```

Replace commit placeholders with actual revisions. The helper restores the recorded order, validates source identity and every chunk hash, and refuses an existing output. Matching token bytes need not produce identical compressed NPZ container bytes. Historical manifests use whole-file hashes, so such a container must not be silently substituted for an archived file.

Complete historical A/B rosters, auxiliary arrays, and every position-level diagnostic are not included. Recovering one retained roster does not reconstruct the entire historical run. [Reproducibility scope](reproducibility.md) explains what the compact evidence can independently recalculate.

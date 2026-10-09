# CNN/DailyMail access and excluded text

The reviewer ZIP contains no CNN/DailyMail articles, token sequences or
individual target-token ranks. The same exclusion now applies to the other
corpora. Numerical measurements, available source identifiers and hashes
remain. Top-k accuracy is rebuilt from exact per-chunk hit rates. See
REDUCTIONS.json and RELEASE_CONTENT_AUDIT.json for the transformations and scan.

The upstream dataset is [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail),
configuration 3.0.0, train split, article field. Cite Hermann et al. (2015)
and See et al. (2017). LICENSE_REVIEW.txt records the published metadata and
the unresolved article-redistribution permission. The package grants no
rights to article text.

DATA_PREPARATION.md documents canonicalization, the 400-character filter,
source IDs, GPT-2 tokenization, non-overlapping 64-token chunks, the two-spaced-
chunk cap per article, source-role separation and chunk-byte hash checks.
It also describes the optional recover_corpus_chunks.py helper for restoring
a retained ordered roster from separately obtained upstream data.

The historical downloader did not pin the dataset revision. Commit
96df5e686bee6baa90b8bee7c28b81fa3fa6223d was found in the local cache, but this
does not establish the revision for every run. Record the commit acquired
and verify source/chunk hashes before claiming exact reconstruction.

Complete correction-study allocations and every auxiliary input are not
delivered. Restoring a reference roster alone does not recreate all forecasts.
TRAINING.md states the additional neural input requirements. No download,
inference or retraining is needed for the default numerical reproduction.
Keep newly acquired text/token files outside the submission archive.

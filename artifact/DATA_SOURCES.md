# Data sources and sampling units

| Corpus | Original publication | Public dataset | Sampling unit used here |
|---|---|---|---|
| TinyStories | Eldan and Li, 2023, arXiv:2305.07759 | https://huggingface.co/datasets/roneneldan/TinyStories | Story |
| WikiText-103 | Merity et al., 2017, Pointer Sentinel Mixture Models | https://huggingface.co/datasets/Salesforce/wikitext | Retained paragraph |
| CNN/DailyMail | Hermann et al., 2015, NeurIPS 28, and See et al., 2017, ACL P17-1099 | https://huggingface.co/datasets/abisee/cnn_dailymail | Article, cased configuration 3.0.0 |
| GPT-2 tokenizer | Radford et al., 2019 | https://huggingface.co/openai-community/gpt2 | 50,257 content-token IDs |

Sizes in this paper count 64-token chunks, not context radius or independent
documents. The reference, auxiliary, A/B and evaluation roles have separate
source allocations. Multiple chunks from a source stay in the same role.
TinyStories is a synthetic corpus of stories generated with GPT-3.5 and GPT-4.

The release contains no corpus text, token sequences or target-token ranks
for any corpus. It retains available source IDs, text/chunk hashes, manifests
and compact numerical measurements. DATA_PREPARATION.md explains acquisition,
source allocation, chunking, masks and verification. Full historical rosters
are not all included. Check hashes before claiming an exact input recovery.

The recorded upstream cards identify CDLA Sharing 1.0 for TinyStories, CC BY-SA
4.0 for WikiText-103, and MIT for the tokenizer. CNN/DailyMail's dataset metadata
declares Apache-2.0, while its licensing paragraph names version 1.0.0. The
original processing repository has an MIT software license. These statements
do not establish permission to redistribute the cased 3.0.0 articles used here.
`LICENSE_REVIEW.txt` records the primary sources, the local cached card revision,
and the distinction between reporting license information and obtaining rights.
`NEWS_DATA_ACCESS.md` gives upstream acquisition and preprocessing instructions.
Removing the news arrays addresses redistribution by this package, but does not
establish the legal basis for research use. No replacement corpus
has been used to produce the reported results. These notes grant no new rights.

Public text may contain personal information, offensive material or biases.
There was no new participant recruitment and this study does not assess fairness.

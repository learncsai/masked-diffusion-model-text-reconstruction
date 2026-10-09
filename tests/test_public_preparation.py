"""Integrity checks for new source allocations using a deterministic fake tokenizer."""
import hashlib
import numpy as np
import pytest
from scripts.prepare_public_data import allocate_roles, save_role, save_training_role, write


class Tokenizer:
    eos_token_id = 50256
    pad_token_id = 50258
    def encode(self,text,add_special_tokens=False):
        return [int(v) for v in text.split()]


def document(index, length=192):
    text = ' '.join(str(index*1000+j) for j in range(length))
    digest = hashlib.sha256(text.encode()).hexdigest()
    return dict(document_id=f'tinystories:{index}:{digest[:16]}',sha256=digest,text=text)


def test_truncated_source_never_crosses_roles_and_news_chunks_are_spaced():
    roles, remaining, _ = allocate_roles(Tokenizer(),[document(i,640) for i in range(8)],
                                        [('reference',3),('a',3),('b',3)],cap=2)
    seen = set()
    for ids, records in roles.values():
        assert ids.shape == (3,64)
        sources = {r['document_id'] for r in records}
        assert not seen & sources
        seen |= sources
        assert [r['chunk_index'] for r in records] == [2,7,2]
    assert not seen & {d['document_id'] for d in remaining}


def test_duplicate_chunks_and_insufficient_sources_do_not_silently_pass():
    first = document(1)
    duplicate = dict(first,document_id='tinystories:2:distinct',sha256='different')
    roles, _, hashes = allocate_roles(Tokenizer(),[first,duplicate,document(3)],
                                     [('a',3),('b',2)])
    assert not {r['sha256'] for r in roles['a'][1]} & {r['sha256'] for r in roles['b'][1]}
    assert len(hashes) == 5
    with pytest.raises(ValueError,match='Insufficient'):
        allocate_roles(Tokenizer(),[document(1)], [('a',4)])


def test_duplicate_canonical_sources_fail_before_allocation():
    d = document(1)
    with pytest.raises(ValueError,match='duplicate sources'):
        allocate_roles(Tokenizer(),[d,dict(d,document_id='other')],[('a',1)])


def test_prepared_input_layout_loads_with_archived_three_corpus_trainer(tmp_path):
    from diffusion_lm_rmt.matched_mdlm import load_inputs
    docs = [document(i,128) for i in range(12)]
    roles, _, _ = allocate_roles(Tokenizer(),docs,[('auxiliary',8),('evaluation',2),
                                                ('reference',2),('pair1_a',2),('pair1_b',2)])
    data = tmp_path/'data/diffusion_llm'
    original = data/'processed_64_wikitext_stats'
    follow = data/'reproducibility_followup_v1/wikitext'
    calibration = data/'calibration_diagnostics_v1/wikitext'
    for source,target in [('auxiliary','replicate3_train_a'),('evaluation','eval')]:
        save_training_role(original,target,roles[source])
    splits = {'eval':{'chunks':roles['evaluation'][1], 'documents':[
        {'document_id':d['document_id'],'sha256':d['sha256']} for d in docs]}}
    write(original/'confirmatory_split_manifest.json',dict(splits=splits))
    for name in ['pair1_a','pair1_b']:
        save_role(follow,name,roles[name])
    import json
    (follow/'raw_documents.jsonl').write_text('\n'.join(json.dumps(d) for d in docs)+'\n')
    save_role(calibration,'reference_original',roles['reference'])
    np.savez_compressed(calibration/'grid_512_512_q0.5.npz',eval_ids=roles['evaluation'][0],
                        mask=np.ones((2,64),dtype=bool))
    write(data/'gpt2_tokenizer/tokenizer_metadata.json',dict(mask_token_id=50257,eos_token_id=50256))
    loaded, mask, _, audit = load_inputs(tmp_path,dict(corpus='wikitext',seed=7,pairs=1,
        corpus_sizes=[2],evaluation_chunks=2,classes=50257,mask_token_id=50257))
    assert len(loaded)==6 and mask.shape==(2,64)
    assert audit['pair1_a']['chunks']==2

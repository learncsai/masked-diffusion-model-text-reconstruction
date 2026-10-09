"""Audit fresh-source separation, forecast timing, and numerical follow-up records."""
from pathlib import Path
import hashlib
import json
import gzip
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(ROOT/'configs/correction_followup_v1.json');out=ROOT/cfg['output']
saved=read(out/'method_saved.json');freeze=read(out/'confirmation_forecasts_saved.json')
analysis=read(out/'confirmation_analysis_saved.json')
assert sha(ROOT/'configs/correction_followup_v1.json')==saved['config_sha256']
assert sha(ROOT/'docs/correction_followup_protocol.txt')==saved['protocol_sha256']
for name,digest in saved['code_sha256'].items():assert sha(ROOT/name)==digest,name
assert analysis['code_sha256']==sha(ROOT/'scripts/summarize_correction_followup.py')
assert analysis['test_sha256']==sha(ROOT/'tests/test_correction_followup.py')
rows=[]
tokenizer=read(ROOT/'data/diffusion_llm/gpt2_tokenizer/tokenizer.json')
vocab=dict(tokenizer['model']['vocab'])
vocab.update({r['content']:r['id'] for r in tokenizer.get('added_tokens',[])})
vocab_hash=hashlib.sha256(json.dumps(sorted(vocab.items(),key=lambda x:x[1]),
    ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
for c in cfg['corpora']:
    old=ROOT/cfg['existing_inputs']/c;new=ROOT/cfg['inputs']/c
    oldaudit=read(old/'source_audit.json');newaudit=read(new/'source_audit.json')
    assert vocab_hash==read(old/'complete.json')['tokenizer_sha256']
    assert sha(old/'fresh_source_pool.jsonl.gz')==read(old/'source_pool.json')['sha256']
    new_ids={r for role in newaudit['roles'].values() for r in role['source_ids']}
    actual_text_hashes={}
    with gzip.open(old/'fresh_source_pool.jsonl.gz','rt',encoding='utf-8') as pool:
        for line in pool:
            raw=json.loads(line)
            if raw['document_id'] in new_ids:
                digest=hashlib.sha256(raw['text'].encode()).hexdigest()
                assert digest==raw['sha256']
                actual_text_hashes[raw['document_id']]=digest
    assert set(actual_text_hashes)==new_ids
    excluded={r for role in oldaudit['roles'].values() for r in role['source_ids']}
    text={r for role in oldaudit['roles'].values() for r in role['text_hashes']}
    chunks=set()
    for d in [old,ROOT/'results/lag_reference_sensitivity_v1/inputs'/c,ROOT/'results/reference_ratio_v1/inputs'/c]:
        for path in d.glob('*.json'):
            data=read(path)
            if isinstance(data,list) and data and isinstance(data[0],dict) and 'document_id' in data[0]:
                excluded.update(r['document_id'] for r in data);chunks.update(r['sha256'] for r in data)
            if path.name=='excluded_sources.json':
                excluded.update(data['source_ids']);text.update(data['text_hashes']);chunks.update(data['chunk_hashes'])
    done=read(new/'complete.json')
    assert saved['saved_utc']<done['saved_utc']
    for name,digest in done['files_sha256'].items():assert sha(new/name)==digest,name
    for role,data in newaudit['roles'].items():
        ids=set(data['source_ids']);texts=set(data['text_hashes'])
        assert texts=={actual_text_hashes[i] for i in ids}
        records=read(new/(role+'.json'));tokens=np.load(new/(role+'.npz'))['ids']
        hashes={r['sha256'] for r in records}
        assert len(hashes)==len(tokens)==2048
        assert not ids&excluded and not texts&text and not hashes&chunks
        assert ids=={r['document_id'] for r in records}
        for token,record in zip(tokens,records):assert hashlib.sha256(token.astype(np.int32).tobytes()).hexdigest()==record['sha256']
        excluded.update(ids);text.update(texts);chunks.update(hashes)
    f=out/'confirmation'/c/'forecasts.npz';o=out/'confirmation'/c/'observed.npz'
    assert sha(f)==freeze['files_sha256'][f'confirmation/{c}/forecasts.npz']
    with np.load(f) as z:
        assert saved['saved_utc']<str(z['saved_utc'])<freeze['saved_utc']
        assert (z['forecasts']>0).all()
    with np.load(o) as z:
        assert max(freeze['saved_utc'],analysis['saved_utc'])<str(z['saved_utc'])
        assert str(z['forecast_freeze_sha256'])==sha(out/'confirmation_forecasts_saved.json')
        np.testing.assert_array_equal(z['observed'],z['per_chunk'].mean(axis=2))
    rows.append(dict(corpus=c,roles=len(newaudit['roles']),new_sources=sum(x['sources'] for x in newaudit['roles'].values()),
                     new_chunks=30720,source_text_chunk_overlap=0,input_manifest_sha256=sha(new/'complete.json')))
result=dict(status='passed',source_audit=rows,all_forecasts_precede_all_outcomes=True,
            analysis_saved_before_outcomes=True,method_and_code_hashes_unchanged=True,
            tokenizer_vocabulary_matches_original=True,raw_pool_and_selected_text_hashes_verified=True,
            tests='Three independent checks: dense output sample variance, whole-source split, unequal-size inverse law.',
            no_neural_training=True,external_preregistration=False)
(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

"""Allocate new source-disjoint reference banks and A/B corpora. No scoring."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import read_json,write_json,sha,utc,validate_records


def prepare(corpus,config_path):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from diffusion_lm_rmt.neural_data import _canonical_text,_chunk_split
    config=read_json(config_path)
    folder=ROOT/config['inputs']/corpus
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        done=read_json(folder/'complete.json')
        assert done['config_sha256']==sha(config_path)
        for rel,digest in done['files_sha256'].items():
            assert sha(folder/rel)==digest,rel
        print(f'Already prepared: {corpus}',flush=True)
        return
    old=ROOT/'results/lag_reference_sensitivity_v1/inputs'/corpus
    recent=ROOT/'results/reference_ratio_v1/inputs'/corpus
    raw_folder=ROOT/'data/diffusion_llm/lag_disagreement_extension_v1'/corpus
    provenance=read_json(raw_folder/'provenance.json')
    assert sha(raw_folder/'raw_documents.jsonl')==provenance['raw_sha256']
    used_texts=set()
    last_index=-1
    with (raw_folder/'raw_documents.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record=json.loads(line)
            used_texts.add(record['sha256'])
            last_index=max(last_index,int(record['document_id'].split(':')[1]))
    excluded=read_json(old/'excluded_sources.json')
    used_texts.update(excluded['text_hashes'])
    historical_sources=set(excluded['source_ids'])
    historical_chunks=set(excluded['chunk_hashes'])
    for directory in (old,recent):
        for path in directory.glob('*.json'):
            records=read_json(path)
            if isinstance(records,list) and records and isinstance(records[0],dict) and 'document_id' in records[0]:
                historical_sources.update(r['document_id'] for r in records)
                historical_chunks.update(r['sha256'] for r in records)
    cache=folder/'fresh_source_pool.jsonl.gz'
    pool_meta=folder/'source_pool.json'
    if cache.exists() and pool_meta.exists():
        meta=read_json(pool_meta)
        assert sha(cache)==meta['sha256']
        with gzip.open(cache,'rt',encoding='utf-8') as source:
            fresh=[json.loads(line) for line in source]
    else:
        settings=provenance['dataset']
        dataset=load_dataset(settings['path'],settings['name'],split=settings['split'],
            revision=provenance['revision'],streaming=True)
        fresh=[]
        claimed=set(used_texts)
        print(f'{corpus}: collecting {config["fresh_source_pool"]} new sources after row {last_index}',flush=True)
        for index,row in enumerate(dataset):
            if index<=last_index:
                continue
            text=_canonical_text(row[settings['text_field']])
            if len(text)<settings['minimum_characters'] or (settings['drop_headings'] and text.startswith('=')):
                continue
            digest=hashlib.sha256(text.encode()).hexdigest()
            if digest in claimed:
                continue
            claimed.add(digest)
            fresh.append(dict(document_id=f'{corpus}:{index}:{digest[:16]}',sha256=digest,text=text))
            if len(fresh)%16000==0:
                print(f'{corpus}: {len(fresh)} fresh sources',flush=True)
            if len(fresh)==config['fresh_source_pool']:
                break
        if len(fresh)!=config['fresh_source_pool']:
            raise RuntimeError(f'Insufficient new sources: {corpus}, {len(fresh)}')
        with gzip.open(cache,'wt',encoding='utf-8',compresslevel=3) as target:
            for record in fresh:
                target.write(json.dumps(record,ensure_ascii=False)+'\n')
        write_json(pool_meta,dict(saved_utc=utc(),sha256=sha(cache),sources=len(fresh),
            historical_last_row=last_index,dataset=settings,revision=provenance['revision'],
            historical_pool_sha256=provenance['raw_sha256'],
            note='New block beyond historical sources, shuffled before assigning roles.'))
    assert len({d['sha256'] for d in fresh})==len(fresh)
    assert not {d['sha256'] for d in fresh}&used_texts
    assert not {d['document_id'] for d in fresh}&historical_sources
    lookup={d['document_id']:d['sha256'] for d in fresh}
    rng=np.random.default_rng(config['seed']+config['corpora'].index(corpus))
    rng.shuffle(fresh)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/diffusion_llm/gpt2_tokenizer',local_files_only=True)
    original=read_json(old/'extension_complete.json')
    vocabulary=hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items(),key=lambda x:x[1]),
        ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert vocabulary==original['tokenizer_sha256']
    max_per_source=original['max_chunks_per_document']
    roles=[(f'reference{r}',config['larger_reference_size']) for r in range(config['reference_banks'])]
    roles += [(f'pair{p}_{arm}',max(config['sizes'])) for p in range(config['pairs']) for arm in ('a','b')]
    assigned_sources=set(historical_sources)
    assigned_chunks=set(historical_chunks)
    audit={}
    for name,count in roles:
        path,records_path=folder/f'{name}.npz',folder/f'{name}.json'
        if path.exists() and records_path.exists():
            with np.load(path) as z:
                tokens=z['ids']
            records=read_json(records_path)
        else:
            if shutil.disk_usage(ROOT).free<2_000_000_000:
                raise RuntimeError('Less than 2 GB free. Preparation stopped safely with completed roles retained.')
            tokens,attention,records=_chunk_split(tokenizer,fresh,count,64,
                forbidden=assigned_chunks,max_chunks_per_document=max_per_source)
            assert len(tokens)==count and bool((attention==1).all()),name
            tokens=tokens.numpy().astype(np.int32)
            np.savez_compressed(path,ids=tokens)
            write_json(records_path,records)
        validate_records(tokens,records)
        assert tokens.shape==(count,64)
        assert np.min(tokens)>=0 and np.max(tokens)<50257
        ids={r['document_id'] for r in records}
        chunks={r['sha256'] for r in records}
        assert not ids&assigned_sources,name
        assert not chunks&assigned_chunks,name
        assert len(chunks)==count
        assigned_sources.update(ids)
        assigned_chunks.update(chunks)
        fresh=[r for r in fresh if r['document_id'] not in ids]
        audit[name]=dict(chunks=count,sources=len(ids),source_ids=sorted(ids),
                        text_hashes=sorted(lookup[s] for s in ids))
        print(f'{corpus}: {name}, {count} chunks, {len(ids)} sources, {len(fresh)} sources remain',flush=True)
    for name in ['context.npz','evaluation_records.json','original_manifest.json','reference_original.npz','reference_original.json']:
        shutil.copyfile(old/name,folder/name)
    write_json(folder/'source_audit.json',dict(roles=audit,source_text_chunk_overlap=0,
        historical_sources=len(historical_sources),historical_chunks=len(historical_chunks),
        source_role_disjoint=True,all_pair_arms_disjoint=True))
    names=[f'{role}{suffix}' for role,_ in roles for suffix in ['.npz','.json']]
    names+=['context.npz','evaluation_records.json','original_manifest.json','source_audit.json','source_pool.json']
    write_json(folder/'complete.json',dict(complete=True,prepared_utc=utc(),corpus=corpus,
        config_sha256=sha(config_path),tokenizer_sha256=vocabulary,max_chunks_per_document=max_per_source,
        files_sha256={name:sha(folder/name) for name in names},
        preparation_code_sha256=sha(__file__),historical_context_sha256=sha(old/'context.npz'),
        remaining_sources=len(fresh),zero_source_text_chunk_overlap=True))
    print(f'Preparation complete: {corpus}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/rare_context_prospective_v1.json')
    parser.add_argument('--corpus',required=True,choices=['tinystories','wikitext','cnn_dailymail'])
    args=parser.parse_args()
    prepare(args.corpus,args.config)

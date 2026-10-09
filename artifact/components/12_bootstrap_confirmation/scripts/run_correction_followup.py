"""Missing-baseline study and fresh-source confirmation, without neural training."""
from __future__ import annotations
import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import read_json,write_json,sha,utc,source_groups,validate_records
from diffusion_lm_rmt import sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import per_sequence_squared_error
from diffusion_lm_rmt.source_moment_forecast import reference_forecasts

CONFIG=ROOT/'configs/correction_followup_v1.json'
PROTOCOL=ROOT/'docs/correction_followup_protocol.txt'
CORE=['src/diffusion_lm_rmt/source_moment_forecast.py','src/diffusion_lm_rmt/sparse_categorical_lag.py',
      'src/diffusion_lm_rmt/categorical_lag.py','src/diffusion_lm_rmt/neural_data.py']
OFFSETS=(-2,-1,1,2)
KW=dict(smoothing=5,ridge=.01)

def initialize():
    cfg=read_json(CONFIG)
    out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    contract=dict(config_sha256=sha(CONFIG),protocol_sha256=sha(PROTOCOL),
                  code_sha256={name:sha(ROOT/name) for name in [*CORE,'scripts/run_correction_followup.py']})
    path=out/'method_saved.json'
    if path.exists():
        old=read_json(path)
        assert all(old[k]==v for k,v in contract.items()),'Frozen follow-up design/code changed'
    else:
        write_json(path,dict(saved_utc=utc(),external_preregistration=False,**contract))
    return cfg,out

def context(folder):
    with np.load(folder/'context.npz') as z:
        return tuple(z[k] for k in ('evaluation','mask','prior','gram','cross'))

def halves(tokens,groups,seed):
    order=np.random.default_rng(seed).permutation(len(groups))
    result=[]
    for selected in np.array_split(order,2):
        indices=np.concatenate([groups[i] for i in selected])
        ends=np.cumsum([0,*[len(groups[i]) for i in selected]])
        local=[np.arange(a,b) for a,b in zip(ends[:-1],ends[1:])]
        result.append((tokens[indices],local,len(selected)/len(groups)))
    return result

def predict(tokens,ctx):
    e,mask,p,gram,cross=ctx
    tables=lag.conditional_tables(lag.pair_counts(tokens,OFFSETS,len(p)),p,5)
    rows,_,pred=lag.masked_prediction(e,mask,tables,p,OFFSETS,gram,cross,.01)
    return rows,pred

def disagreement(a,b,ctx):
    rows,pa=predict(a,ctx);_,pb=predict(b,ctx)
    d=lag.squared_difference(pa,pb)
    return per_sequence_squared_error(rows,d,len(ctx[0]))

def baselines(corpus):
    cfg,out=initialize();ci=cfg['corpora'].index(corpus)
    old=read_json(ROOT/'configs/rare_context_prospective_v1.json')
    folder=ROOT/cfg['existing_inputs']/corpus;ctx=context(folder)
    e,mask,p,gram,cross=ctx
    dest=out/'baselines'/corpus;dest.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    for bank in range(6):
        tokens=np.load(folder/f'reference{bank}.npz')['ids']
        records=read_json(folder/f'reference{bank}.json')
        for m in old['reference_sizes']:
            groups=source_groups(records[:m]);full=tokens[:m]
            partition=halves(full,groups,old['seed']+10000*ci+100*bank+m)
            direct=dest/f'direct_bank{bank}_m{m}.npz'
            if not direct.exists():
                a,b=[h[0] for h in partition]
                value=disagreement(a,b,ctx)
                np.savez_compressed(direct,per_chunk=value,half_chunks=[len(a),len(b)],
                                    harmonic_half_size=2/(1/len(a)+1/len(b)),saved_utc=utc())
            for n in old['sizes']:
                path=dest/f'bootstrap_bank{bank}_m{m}_n{n}.npz'
                if path.exists(): continue
                batches=np.zeros((3,cfg['bootstrap_batches']))
                for component,(sample,local,_) in enumerate([(full,groups,1.),*partition]):
                    for batch in range(cfg['bootstrap_batches']):
                        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],ci,bank,m,n,component,batch]))
                        d,_=lag.bootstrap_disagreement(sample,local,e,mask,p,OFFSETS,gram,cross,
                            n=n,**KW,draws=cfg['bootstrap_draws_per_batch'],law='source_cluster',rng=rng)
                        batches[component,batch]=d.mean()
                fraction=np.array([h[2] for h in partition])
                jk=2*batches[0]-fraction@batches[1:]
                assert np.all(np.isfinite(batches)) and (batches>0).all()
                np.savez_compressed(path,batches=batches,jackknife_batches=jk,
                    source_fractions=fraction,raw_jackknife=jk.mean(),
                    forecasts=[batches[0].mean(),max(0.,jk.mean())],saved_utc=utc(),
                    method_sha256=sha(out/'method_saved.json'))
                print(f'{corpus} bank {bank+1}/6 m={m} n={n}: {time.perf_counter()-started:.0f}s',flush=True)

def prepare(corpus):
    from transformers import AutoTokenizer
    from diffusion_lm_rmt.neural_data import _chunk_split
    cfg,out=initialize();design=cfg['confirmatory']
    old=ROOT/cfg['existing_inputs']/corpus
    folder=ROOT/cfg['inputs']/corpus;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        done=read_json(folder/'complete.json')
        assert done['method_sha256']==sha(out/'method_saved.json')
        for name,digest in done['files_sha256'].items(): assert sha(folder/name)==digest
        return
    done=read_json(old/'complete.json');used=set();used_text=set();used_chunks=set()
    audit=read_json(old/'source_audit.json')
    for role in audit['roles'].values():
        used.update(role['source_ids']);used_text.update(role['text_hashes'])
    # Include auxiliary/evaluation and every historical allocation in chunk exclusions.
    for directory in [old,ROOT/'results/lag_reference_sensitivity_v1/inputs'/corpus,
                      ROOT/'results/reference_ratio_v1/inputs'/corpus]:
        for path in directory.glob('*.json'):
            data=read_json(path)
            if isinstance(data,list) and data and isinstance(data[0],dict) and 'document_id' in data[0]:
                used.update(r['document_id'] for r in data);used_chunks.update(r['sha256'] for r in data)
            if path.name=='excluded_sources.json':
                used.update(data['source_ids']);used_text.update(data['text_hashes']);used_chunks.update(data['chunk_hashes'])
    with gzip.open(old/'fresh_source_pool.jsonl.gz','rt',encoding='utf-8') as f:
        pool=[json.loads(line) for line in f]
    pool=[r for r in pool if r['document_id'] not in used and r['sha256'] not in used_text]
    assert len(pool)==done['remaining_sources'],(corpus,len(pool),done['remaining_sources'])
    np.random.default_rng(cfg['seed']+cfg['corpora'].index(corpus)).shuffle(pool)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/diffusion_llm/gpt2_tokenizer',local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    roles=[(f'reference{b}',max(design['reference_sizes'])) for b in range(design['reference_banks'])]
    roles += [(f'pair{p}_{a}',max(design['sizes'])) for p in range(design['pairs']) for a in ('a','b')]
    new_audit={};files=[]
    for name,count in roles:
        path=folder/f'{name}.npz';roster=folder/f'{name}.json'
        if path.exists() and roster.exists():
            tokens=np.load(path)['ids'];records=read_json(roster)
        else:
            tokens,attention,records=_chunk_split(tokenizer,pool,count,64,forbidden=used_chunks,
                                                 max_chunks_per_document=done['max_chunks_per_document'])
            assert len(tokens)==count and bool((attention==1).all()),'Insufficient unused chunks'
            tokens=tokens.numpy().astype(np.int32)
            np.savez_compressed(path,ids=tokens);write_json(roster,records)
        validate_records(tokens,records)
        ids={r['document_id'] for r in records};chunks={r['sha256'] for r in records}
        text={r['sha256'] for r in pool if r['document_id'] in ids}
        assert not ids&used and not chunks&used_chunks and not text&used_text
        used.update(ids);used_chunks.update(chunks);used_text.update(text)
        pool=[r for r in pool if r['document_id'] not in ids]
        new_audit[name]=dict(sources=len(ids),chunks=count,source_ids=sorted(ids),text_hashes=sorted(text))
        files.extend([path,roster])
        print(f'{corpus}: {name} prepared, {len(pool)} unused sources remain',flush=True)
    shutil.copyfile(old/'context.npz',folder/'context.npz');files.append(folder/'context.npz')
    write_json(folder/'source_audit.json',dict(roles=new_audit,source_text_chunk_overlap=0,
        excluded_old_roles=True,remaining_sources=len(pool)))
    files.append(folder/'source_audit.json')
    write_json(folder/'complete.json',dict(saved_utc=utc(),method_sha256=sha(out/'method_saved.json'),
        files_sha256={p.name:sha(p) for p in files},old_complete_sha256=sha(old/'complete.json')))

def forecast(corpus):
    cfg,out=initialize();d=cfg['confirmatory'];ci=cfg['corpora'].index(corpus)
    folder=ROOT/cfg['inputs']/corpus;done=read_json(folder/'complete.json')
    for name,digest in done['files_sha256'].items(): assert sha(folder/name)==digest
    ctx=context(folder);e,mask,p,gram,cross=ctx
    dest=out/'confirmation'/corpus;dest.mkdir(parents=True,exist_ok=True)
    path=dest/'forecasts.npz'
    if path.exists(): return
    values=np.zeros((d['reference_banks'],len(d['reference_sizes']),len(d['sizes']),4))
    for bank in range(d['reference_banks']):
        tokens=np.load(folder/f'reference{bank}.npz')['ids'];rec=read_json(folder/f'reference{bank}.json')
        for mi,m in enumerate(d['reference_sizes']):
            groups=source_groups(rec[:m]);parts=halves(tokens[:m],groups,cfg['seed']+10000*ci+100*bank+m)
            components=[reference_forecasts(t,g,e,mask,p,OFFSETS,gram,cross,d['sizes'],**KW)
                        for t,g,_ in [(tokens[:m],groups,1.),*parts]]
            for ni,n in enumerate(d['sizes']):
                scalar=np.asarray([c[n]['point_forecasts']@c[n]['point_weight'] for c in components])
                corrected=2*scalar[0]-sum(h[2]*s for h,s in zip(parts,scalar[1:]))
                values[bank,mi,ni]=[scalar[0,2],corrected[2],scalar[0,0],corrected[0]]
    assert np.all(values>0)
    np.savez_compressed(path,forecasts=values,saved_utc=utc(),method_sha256=sha(out/'method_saved.json'),
                        input_sha256=sha(folder/'complete.json'))
    print(f'{corpus}: confirmation forecasts saved before scoring',flush=True)

def freeze():
    cfg,out=initialize();path=out/'confirmation_forecasts_saved.json'
    files={f'confirmation/{c}/forecasts.npz':sha(out/'confirmation'/c/'forecasts.npz') for c in cfg['corpora']}
    if path.exists():
        assert read_json(path)['files_sha256']==files
    else:
        assert not list((out/'confirmation').glob('*/observed.npz'))
        write_json(path,dict(saved_utc=utc(),files_sha256=files,method_sha256=sha(out/'method_saved.json'),
                            external_preregistration=False))
    return cfg,out,path

def score(corpus):
    cfg,out,freeze_path=freeze();d=cfg['confirmatory']
    folder=ROOT/cfg['inputs']/corpus;ctx=context(folder)
    path=out/'confirmation'/corpus/'observed.npz'
    if path.exists(): return
    values=[]
    for pair in range(d['pairs']):
        arms=[np.load(folder/f'pair{pair}_{a}.npz')['ids'] for a in ('a','b')]
        values.append([disagreement(arms[0][:n],arms[1][:n],ctx) for n in d['sizes']])
    np.savez_compressed(path,per_chunk=np.asarray(values),observed=np.asarray(values).mean(axis=2),
                        saved_utc=utc(),forecast_freeze_sha256=sha(freeze_path))
    print(f'{corpus}: six new pairs scored',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['initialize','prepare','forecast','freeze','score','baselines'])
    ap.add_argument('--corpus',choices=['tinystories','wikitext','cnn_dailymail'])
    a=ap.parse_args()
    if a.action in ['initialize','freeze']: globals()[a.action]()
    else: globals()[a.action](a.corpus)

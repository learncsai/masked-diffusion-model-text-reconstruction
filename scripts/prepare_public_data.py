"""Prepare a new, source-disjoint public-data replication. Never replay old hashes."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import numpy as np
from diffusion_lm_rmt.neural_data import CORPORA, _canonical_text, _chunk_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def allocate_roles(tokenizer, documents, budgets, cap=None, forbidden=None):
    """Allocate entire sources in order and exclude exact duplicate chunks across roles."""
    pool = list(documents)
    if len({d['document_id'] for d in pool}) != len(pool) or len({d['sha256'] for d in pool}) != len(pool):
        raise ValueError('Preparation pool contains duplicate sources or canonical texts')
    claimed = set(forbidden or ())
    roles = {}
    for name, budget in budgets:
        if budget < 1:
            raise ValueError('Chunk budgets must be positive')
        ids, attention, records = _chunk_split(tokenizer, pool, budget, 64,
            forbidden=claimed, max_chunks_per_document=cap)
        if len(ids) != budget or not bool((attention == 1).all()):
            raise ValueError(f'Insufficient full chunks for {name}: need {budget}, got {len(ids)}')
        used = {r['document_id'] for r in records}
        claimed.update(r['sha256'] for r in records)
        roles[name] = (ids.numpy().astype('<i4'), records)
        pool = [d for d in pool if d['document_id'] not in used]
    return roles, pool, claimed


def save_role(folder, name, role):
    folder.mkdir(parents=True, exist_ok=True)
    ids, records = role
    np.savez_compressed(folder/f'{name}.npz', ids=ids)
    write(folder/f'{name}.json', records)


def save_training_role(folder, name, role):
    import torch
    ids, records = role
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(input_ids=torch.as_tensor(ids.astype(np.int64)),
        attention_mask=torch.ones(ids.shape,dtype=torch.long),chunk_records=records),folder/f'{name}.pt')


def source_audit(roles, documents):
    lookup = {d['document_id']: d['sha256'] for d in documents}
    return dict(roles={name: dict(chunks=len(ids), sources=len({r['document_id'] for r in records}),
        source_ids=sorted({r['document_id'] for r in records}),
        text_hashes=sorted({lookup[r['document_id']] for r in records}))
        for name, (ids, records) in roles.items()},
        source_role_disjoint=True, all_pair_arms_disjoint=True, source_text_chunk_overlap=0)


def prepare(corpus, dataset_revision=None, tokenizer_revision=None):
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from diffusion_lm_rmt.categorical_lag import source_groups
    from diffusion_lm_rmt import sparse_categorical_lag as lag
    cfg = json.loads((ROOT/'configs/rare_context_prospective_v1.json').read_text())
    neural = json.loads((ROOT/'configs/matched_mdlm_tinystories_pilot_v1.json').read_text())
    dest = ROOT/cfg['inputs']/corpus
    processed_names = dict(tinystories='processed_64_tinystories_matched',
        wikitext='processed_64_wikitext_stats', cnn_dailymail='processed_64_cnn_dailymail_capped')
    original = ROOT/'data/diffusion_llm'/processed_names[corpus]
    follow = ROOT/'data/diffusion_llm/reproducibility_followup_v1'/corpus
    calibration = ROOT/'data/diffusion_llm/calibration_diagnostics_v1'/corpus
    if dest.exists() or original.exists() or follow.exists() or calibration.exists():
        raise FileExistsError('Inputs already exist. Use a fresh clone/worktree for a new replication.')
    settings = CORPORA[corpus]
    api = HfApi()
    dataset_commit = api.dataset_info(settings['path'], revision=dataset_revision).sha
    tokenizer_commit = api.model_info('openai-community/gpt2', revision=tokenizer_revision).sha
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2', revision=tokenizer_commit)
    tokenizer.add_special_tokens({'mask_token':'[MASK]', 'pad_token':'[PAD]'})
    assert (tokenizer.eos_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id) == (50256,50257,50258)
    tokenizer_dir = ROOT/'data/diffusion_llm/gpt2_tokenizer'
    ordered = sorted(tokenizer.get_vocab().items(), key=lambda x:x[1])
    metadata = dict(base_model='openai-community/gpt2', vocab_size=len(tokenizer),
        mask_token_id=50257, pad_token_id=50258, eos_token_id=50256,
        vocabulary_sha256=hashlib.sha256(json.dumps(ordered,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
    if (tokenizer_dir/'tokenizer_metadata.json').exists():
        if json.loads((tokenizer_dir/'tokenizer_metadata.json').read_text()) != metadata:
            raise ValueError('Tokenizer differs from the existing corpus preparation')
        revisions_path = tokenizer_dir/'acquisition.json'
        if json.loads(revisions_path.read_text())['tokenizer_commit'] != tokenizer_commit:
            raise ValueError('Use the same immutable tokenizer commit for every corpus')
    else:
        tokenizer.save_pretrained(tokenizer_dir)
        write(tokenizer_dir/'tokenizer_metadata.json', metadata)
        write(tokenizer_dir/'acquisition.json', dict(tokenizer_commit=tokenizer_commit))
    stream = load_dataset(settings['path'], settings['name'], split='train',
        revision=dataset_commit, streaming=True)
    documents, seen = [], set()
    required = 64000 + cfg['fresh_source_pool']
    print(f'{corpus}: acquiring {required} unique sources at {dataset_commit}', flush=True)
    for index, row in enumerate(stream):
        text = _canonical_text(row[settings['text_field']])
        if len(text) < settings['minimum_characters'] or (settings['drop_headings'] and text.startswith('=')):
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        documents.append(dict(document_id=f'{corpus}:{index}:{digest[:16]}', sha256=digest, text=text))
        if len(documents) % 16000 == 0:
            print(f'{corpus}: {len(documents)}/{required} sources', flush=True)
        if len(documents) == required:
            break
    if len(documents) != required:
        raise ValueError(f'Not enough upstream sources: {len(documents)} of {required}')
    base, fresh = documents[:64000], documents[64000:]
    ci = cfg['corpora'].index(corpus)
    ordered_base = list(base)
    np.random.default_rng(neural['seed']+ci).shuffle(ordered_base)
    cap = 2 if corpus == 'cnn_dailymail' else None
    budgets = [('auxiliary',4096), ('evaluation',128), ('reference',2048)]
    budgets += [(f'pair{p}_{a}',8192) for p in (1,2,3) for a in ('a','b')]
    roles, _, claimed = allocate_roles(tokenizer, ordered_base, budgets, cap)
    original.mkdir(parents=True)
    splits = {}
    for name, output in [('auxiliary','replicate3_train_a'), ('evaluation','eval')]:
        ids, records = roles[name]
        save_training_role(original,output,(ids,records))
        used = {r['document_id'] for r in records}
        splits[output] = dict(chunks=records, documents=[{k:d[k] for k in ['document_id','sha256']}
                                                       for d in base if d['document_id'] in used])
    write(original/'confirmatory_split_manifest.json',dict(tokenizer=metadata,splits=splits,
        max_chunks_per_document=cap, replication='New public-data allocation; not historical input recovery'))
    follow.mkdir(parents=True)
    with (follow/'raw_documents.jsonl').open('w',encoding='utf-8') as output:
        for doc in base:
            output.write(json.dumps(doc,ensure_ascii=False)+'\n')
    for name, role in roles.items():
        if name.startswith('pair'):
            save_role(ROOT/'data/matched_extensions'/corpus,name,role)
            save_role(follow,name,(role[0][:2048],role[1][:2048]))
    save_role(calibration,'reference_original',roles['reference'])
    aux_ids, aux_records = roles['auxiliary']
    groups = source_groups(aux_records)
    np.random.default_rng(neural['seed']).shuffle(groups)
    train = aux_ids[np.concatenate(groups[:len(groups)//2])]
    val = aux_ids[np.concatenate(groups[len(groups)//2:])]
    prior = np.bincount(train.ravel(),minlength=50257).astype(float)
    prior /= prior.sum()
    gram, cross = lag.fitting_equations(train,val,prior,(-2,-1,1,2),5,.5,
                                       np.random.default_rng(neural['seed']+1))
    evaluation = roles['evaluation'][0]
    mask = np.random.default_rng(neural['seed']+17+ci).random(evaluation.shape) < .5
    mask[~mask.any(axis=1),0] = True
    np.savez_compressed(calibration/'grid_512_512_q0.5.npz',eval_ids=evaluation,mask=mask)
    ordered_fresh = list(fresh)
    np.random.default_rng(cfg['seed']+ci).shuffle(ordered_fresh)
    budgets = [(f'reference{b}',cfg['larger_reference_size']) for b in range(cfg['reference_banks'])]
    budgets += [(f'pair{p}_{a}',max(cfg['sizes'])) for p in range(cfg['pairs']) for a in ('a','b')]
    new_roles, remaining, _ = allocate_roles(tokenizer,ordered_fresh,budgets,cap,claimed)
    for name, role in new_roles.items():
        save_role(dest,name,role)
    save_role(dest,'reference_original',roles['reference'])
    write(dest/'evaluation_records.json',roles['evaluation'][1])
    np.savez_compressed(dest/'context.npz',evaluation=evaluation,mask=mask,prior=prior,gram=gram,cross=cross)
    inputs = {p.relative_to(ROOT).as_posix():sha(p) for p in [original/'replicate3_train_a.pt',original/'eval.pt',
               original/'confirmatory_split_manifest.json',tokenizer_dir/'tokenizer_metadata.json']}
    neural.update(corpus=corpus)
    write(dest/'original_manifest.json',dict(protocol=dict(config=neural,inputs=inputs)))
    write(dest/'source_audit.json',source_audit(new_roles,fresh))
    with gzip.open(dest/'fresh_source_pool.jsonl.gz','wt',encoding='utf-8') as output:
        for doc in fresh:
            output.write(json.dumps(doc,ensure_ascii=False)+'\n')
    write(dest/'source_pool.json',dict(dataset=settings,dataset_commit=dataset_commit,
        tokenizer_commit=tokenizer_commit,sources=len(fresh),replication='New public-data experiment'))
    write(dest/'complete.json',dict(complete=True,corpus=corpus,config_sha256=sha(ROOT/'configs/rare_context_prospective_v1.json'),
        tokenizer_sha256=metadata['vocabulary_sha256'],max_chunks_per_document=cap,
        remaining_sources=len(remaining),files_sha256={p.name:sha(p) for p in dest.iterdir() if p.is_file()},
        zero_source_text_chunk_overlap=True,replication='New public-data allocation; historical outputs will differ'))
    write(ROOT/'data/acquisition'/f'{corpus}.json',dict(dataset=settings,dataset_commit=dataset_commit,
        tokenizer_commit=tokenizer_commit,source_count=required,preparation_sha256=sha(__file__),
        source_audit=source_audit(roles,base),versions=dict(torch=torch.__version__,numpy=np.__version__)))
    print(f'{corpus}: prepared matched and 20-pair correction inputs. New replication, not historical replay.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', choices=list(CORPORA), required=True)
    parser.add_argument('--dataset-revision', help='Optional commit/ref; resolved and recorded as an immutable commit.')
    parser.add_argument('--tokenizer-revision', help='Optional GPT-2 commit/ref; resolved and recorded as an immutable commit.')
    args = parser.parse_args()
    prepare(args.corpus,args.dataset_revision,args.tokenizer_revision)

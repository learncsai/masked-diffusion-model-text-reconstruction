"""Recover a retained chunk roster from a separately acquired upstream corpus.

Optional data acquisition, never called by reproduce.py. Does not train models.
Requires datasets and transformers as additional dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

DATASETS={
    'tinystories':('roneneldan/TinyStories',None,'text'),
    'wikitext':('Salesforce/wikitext','wikitext-103-raw-v1','text'),
    'cnn_dailymail':('abisee/cnn_dailymail','3.0.0','article')}

def recover(records,articles,tokenizer,corpus):
    wanted={}
    for row in records:
        name,index,prefix=row['document_id'].split(':')
        if name!=corpus:raise ValueError('Roster contains a different corpus')
        wanted.setdefault(int(index),[]).append(row)
    keys={(r['document_id'],int(r['chunk_index'])) for r in records}
    if not keys:raise ValueError('Empty roster')
    chunks={}
    for source_index,article in articles:
        if source_index not in wanted:continue
        text=' '.join(article.replace('\r','\n').split())
        digest=hashlib.sha256(text.encode('utf-8')).hexdigest()
        actual_id=f'{corpus}:{source_index}:{digest[:16]}'
        if {r['document_id'] for r in wanted[source_index]}!={actual_id}:
            raise ValueError(f'Upstream text or ordering differs at source {source_index}')
        tokens=tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]
        for row in wanted[source_index]:
            start=int(row['chunk_index'])*64
            chunk=np.asarray(tokens[start:start+64],dtype='<i4')
            if len(chunk)!=64 or hashlib.sha256(chunk.tobytes()).hexdigest()!=row['sha256']:
                raise ValueError(f'Chunk/tokenizer mismatch: {row["document_id"]}, chunk {row["chunk_index"]}')
            chunks[(row['document_id'],int(row['chunk_index']))]=chunk
        if len(chunks)==len(keys):break
    missing=keys-set(chunks)
    if missing:raise ValueError(f'{len(missing)} required chunks absent from acquired stream')
    return np.stack([chunks[(r['document_id'],int(r['chunk_index']))] for r in records])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',choices=list(DATASETS),required=True)
    p.add_argument('--records',type=Path,required=True,help='Ordered chunk-roster JSON')
    p.add_argument('--revision',required=True,help='Dataset commit to acquire')
    p.add_argument('--tokenizer-revision',required=True,help='GPT-2 tokenizer commit to acquire')
    p.add_argument('--output',type=Path,required=True,help='New .npz outside the submission package')
    args=p.parse_args()
    if args.output.suffix!='.npz':raise ValueError('--output must end in .npz')
    report_path=args.output.with_suffix('.verification.json')
    if args.output.exists() or report_path.exists():raise FileExistsError('Choose new output paths')
    from datasets import load_dataset
    from transformers import AutoTokenizer
    metadata=json.loads((Path(__file__).parent/'docs/tokenizer_metadata.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(metadata['base_model'],revision=args.tokenizer_revision)
    tokenizer.add_special_tokens({'mask_token':'[MASK]','pad_token':'[PAD]'})
    ordered=sorted(tokenizer.get_vocab().items(),key=lambda item:item[1])
    vocab_hash=hashlib.sha256(json.dumps(ordered,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if vocab_hash!=metadata['vocabulary_sha256']:raise ValueError('Tokenizer vocabulary mismatch')
    for key in ['mask_token_id','pad_token_id','eos_token_id']:
        if getattr(tokenizer,key)!=metadata[key]:raise ValueError(f'Tokenizer {key} mismatch')
    records=json.loads(args.records.read_text(encoding='utf-8'))
    dataset,config,field=DATASETS[args.corpus]
    stream=load_dataset(dataset,config,split='train',streaming=True,revision=args.revision)
    ids=recover(records,((i,row[field]) for i,row in enumerate(stream)),tokenizer,args.corpus)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,ids=ids)
    report=dict(status='passed',chunks=len(ids),all_recorded_chunk_hashes_match=True,
        dataset=dataset,configuration=config,split='train',revision_acquired=args.revision,
        tokenizer_revision=args.tokenizer_revision,vocabulary_sha256=vocab_hash,
        scope='Recorded allocation recovered in original order. Chunk-byte identity checked. NPZ container bytes can differ.')
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()

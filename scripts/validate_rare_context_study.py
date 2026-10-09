"""Independent provenance and numerical checks for the correction study."""
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import read_json,write_json,sha,source_groups
from scripts.evaluate_lag_disagreement_extension import load_context
from diffusion_lm_rmt.source_moment_forecast import reference_forecasts


def main():
    config=read_json(ROOT/'configs/rare_context_prospective_v1.json')
    out=ROOT/config['output']
    checks=[]
    for corpus in config['corpora']:
        old=ROOT/'results/lag_reference_sensitivity_v1/inputs'/corpus
        (evaluation,mask,prior,gram,cross),reference,records,_,_=load_context(old)
        started=time.perf_counter()
        result=reference_forecasts(reference,source_groups(records),evaluation,mask,prior,
            (-2,-1,1,2),gram,cross,[2048],smoothing=5.,ridge=.01)[2048]
        legacy=ROOT/'results/reference_ratio_v1'/corpus/'nested_m2048_n2048_forecasts.npz'
        with np.load(legacy) as z:
            expected=z['forecasts']
        np.testing.assert_allclose(result['forecasts'],expected,rtol=2e-10,atol=1e-12)
        checks.append(dict(corpus=corpus,check='source-moment implementation agrees with historical source-loop results',
            max_absolute_difference=float(np.max(np.abs(result['forecasts']-expected))),
            legacy_sha256=sha(legacy),runtime_seconds=time.perf_counter()-started))
        folder=ROOT/config['inputs']/corpus
        done=read_json(folder/'complete.json')
        for name,digest in done['files_sha256'].items():
            assert sha(folder/name)==digest,name
        audit=read_json(folder/'source_audit.json')
        all_sources,all_texts,all_chunks=set(),set(),set()
        for role,entry in audit['roles'].items():
            records=read_json(folder/f'{role}.json')
            sources={r['document_id'] for r in records}
            texts=set(entry['text_hashes'])
            chunks={r['sha256'] for r in records}
            assert sources==set(entry['source_ids'])
            assert len(sources)==entry['sources'] and len(chunks)==entry['chunks']
            assert not sources&all_sources and not texts&all_texts and not chunks&all_chunks
            all_sources.update(sources);all_texts.update(texts);all_chunks.update(chunks)
        checks.append(dict(corpus=corpus,check='all six reference banks and forty A/B arms are source/text/chunk disjoint',
            roles=len(audit['roles']),sources=len(all_sources),chunks=len(all_chunks),
            source_text_chunk_overlap=0))
    write_json(out/'implementation_validation.json',dict(status='passed',checks=checks,
        no_new_ab_outcomes_read=True,validator_sha256=sha(__file__)))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()

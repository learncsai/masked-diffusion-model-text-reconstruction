"""Independent arithmetic and provenance checks for the added comparison."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--check-inputs', action='store_true')
    args=ap.parse_args(); cfg=read(ROOT/'configs/bootstrap_confirmation_v1.json')
    out=ROOT/cfg['output']; saved=read(out/'method_saved.json'); summary=read(out/'summary.json')
    missing=[]
    for name, expected in saved['files_sha256'].items():
        path=ROOT/name
        if path.exists(): assert digest(path)==expected, name
        else: missing.append(name)
    if args.check_inputs: assert not missing, missing
    records=list(csv.DictReader((out/'bank_records.csv').open(newline='',encoding='utf-8')))
    assert len(records)==81
    for j,method in enumerate(summary['methods']):
        values=[]
        for corpus in cfg['corpora']:
            with np.load(ROOT/f'results/correction_followup_v1/confirmation/{corpus}/observed.npz') as z:
                observed=math.fsum(float(v) for v in z['observed'][:,0])/6
            selected=[r for r in records if r['method']==method and r['corpus']==corpus]
            assert {int(r['bank']) for r in selected}=={0,1,2}
            for row in selected:
                assert math.isclose(float(row['observed_mean']),observed,rel_tol=1e-13)
                error=abs(math.log(float(row['forecast'])/observed))
                assert math.isclose(error,float(row['absolute_log_error']),abs_tol=1e-14)
                values.append(error)
        assert math.isclose(math.fsum(values)/9,summary['error'][j],rel_tol=1e-13)
    # Reference rosters retain whole source identities and the original partitions.
    for ci,corpus in enumerate(cfg['corpora']):
        folder=ROOT/cfg['inputs']/corpus
        complete=read(folder/'complete.json')
        if args.check_inputs:
            for name, expected in complete['files_sha256'].items():
                assert digest(folder/name)==expected, (corpus,name)
        audit=read(folder/'source_audit.json')
        occupied=set()
        for role in audit['roles'].values():
            ids=set(role['source_ids']); assert not ids&occupied
            occupied.update(ids)
        for bank in range(3):
            roster=read(folder/f'reference{bank}.json')[:cfg['m']]
            lengths={}
            for row in roster: lengths[row['document_id']]=lengths.get(row['document_id'],0)+1
            order=np.random.default_rng(cfg['partition_seed']+10000*ci+100*bank+cfg['m']).permutation(len(lengths))
            pieces=np.array_split(order,2)
            chunks=np.array(list(lengths.values()))
            with np.load(out/corpus/f'bank{bank}.npz') as z:
                np.testing.assert_array_equal(z['source_counts'],[len(lengths), *map(len,pieces)])
                np.testing.assert_array_equal(z['chunk_counts'],[cfg['m'], *[chunks[p].sum() for p in pieces]])
                assert saved['saved_utc'] < str(z['saved_utc'])
                assert str(z['method_sha256'])==digest(out/'method_saved.json')
            with np.load(ROOT/f'results/correction_followup_v1/confirmation/{corpus}/observed.npz') as z:
                assert str(z['saved_utc']) < saved['saved_utc'], 'Added comparison must retain known-outcome status'
    report=dict(status='passed',banks=9,bank_method_records=81,methods=9,
                source_partition_and_role_checks=True,settings_precede_new_bootstrap=True,
                outcomes_already_known=True,local_token_hashes_checked=args.check_inputs,
                omitted_inputs=missing)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()

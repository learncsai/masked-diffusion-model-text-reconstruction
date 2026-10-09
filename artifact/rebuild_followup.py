"""Rebuild follow-up inference and manuscript rows from compact scalar records."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

LABELS={
    'source_full':'Full source','source_jackknife':'Source jackknife',
    'multinomial':'Multinomial','multinomial_jackknife':'Multinomial jackknife',
    'source_bootstrap':'Source bootstrap','bootstrap_jackknife':'Bootstrap jackknife',
    'direct_inverse_size':r'Direct split, $1/n$','direct_constant':'Direct split, constant',
    'source_times_1.25':r'Source $\times1.25$','multinomial_times_1.25':r'Multinomial $\times1.25$',
    'source_gated':r'Source jackknife, $m\le n$','multinomial_gated':r'Multinomial jackknife, $m\le n$'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)
    else:assert a==b,(a,b)

def render(root):
    result=root/'results/correction_followup_v1'
    r=read(result/'summary.json');c=read(result/'confirmation_summary.json')
    out=result/'rebuilt';out.mkdir(exist_ok=True)
    tables={}
    rows=[];intervals=[]
    for j,method in enumerate(r['methods']):
        if 'gated' not in method:
            rows.append(LABELS[method]+f" & {r['primary']['error'][j]:.3f} & {r['full_grid']['error'][j]:.3f}")
        vals=[]
        for key in ['primary','full_grid','m_gt_n']:
            x=r[key]['error'][j];lo,hi=r[key]['interval'][j]
            vals.append(f'{x:.3f} [{lo:.3f}, {hi:.3f}]')
        intervals.append(LABELS[method]+' & '+' & '.join(vals))
    tables['tab:correction-primary']=rows
    tables['tab:followup-intervals']=intervals
    rows=[]
    for j,method in enumerate(c['methods']):
        vals=[]
        for key in ['primary','full_grid']:
            x=c[key]['error'][j];lo,hi=c[key]['interval'][j]
            vals.append(f'{x:.3f} [{lo:.3f}, {hi:.3f}]')
        rows.append(LABELS[method]+' & '+' & '.join(vals))
    tables['tab:confirmation']=rows
    rows=[]
    for method in r['methods']:
        x=next(x for x in r['decisions'] if x['method']==method and x['target']=='original')
        rows.append(LABELS[method]+' & '+' & '.join(str(x[k]) for k in
            ['success','failure','abstain','successes_at_grid_min','successes_above_grid_min']))
    tables['tab:correction-decisions']=rows
    rows=[]
    for method in r['methods']:
        vals=[]
        for target in ['0.02','0.025','0.03']:
            x=next(x for x in r['decisions'] if x['method']==method and x['target']==target)
            vals.append('/'.join(str(x[k]) for k in ['success','failure','abstain']))
        rows.append(LABELS[method]+' & '+' & '.join(vals))
    tables['tab:decision-targets']=rows
    for label,rows in tables.items():
        (out/(label.replace(':','_')+'.tex')).write_text('\n'.join(rows)+'\n',encoding='utf-8')
    (out/'tables.json').write_text(json.dumps(tables,indent=2)+'\n')
    return tables

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--render-only',action='store_true')
    a=p.parse_args();root=a.root
    if not a.render_only:
        result=root/'results/correction_followup_v1'
        old=read(result/'summary.json');confirm=read(result/'confirmation_summary.json')
        spec=importlib.util.spec_from_file_location('followup',root/'scripts/summarize_correction_followup.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.existing(root,result);mod.confirmation(root,result)
        compare(old,read(result/'summary.json'));compare(confirm,read(result/'confirmation_summary.json'))
    tables=render(root)
    print(f'Follow-up verified: {len(tables)} manuscript tables.')

if __name__=='__main__':main()

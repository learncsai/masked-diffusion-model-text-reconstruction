"""Offline numerical reproduction of the submission through Appendix E."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduced'
NAMES={'tinystories':'TinyStories','wikitext':'WikiText-103','cnn_dailymail':'CNN/DailyMail'}

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def run(folder,args,name):
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONUTF8='1')
    started=time.perf_counter()
    with (OUT/'logs'/f'{name}.txt').open('w',encoding='utf-8') as log:
        proc=subprocess.run([sys.executable,*args],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'{name} failed. Read reproduced/logs/{name}.txt')
    print(f'PASS {name} ({time.perf_counter()-started:.1f}s)',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extract-only',action='store_true',help='Verify hashes and make an inspectable working copy, without analysis.')
    args=ap.parse_args();started=time.perf_counter()
    for name,expected in read(ROOT/'PACKAGE_SHA256.json').items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for name in ['logs','tables','figures','diagnostics']:(OUT/name).mkdir(parents=True,exist_ok=True)
    expected=read(ROOT/'EXPECTED_RESULTS.json')
    work=OUT/('work_'+time.strftime('%Y%m%d_%H%M%S'));shutil.copytree(ROOT/'components',work)
    if args.extract_only:
        print(f'Verified numerical records: {work}');return
    folders={p.name:p for p in work.iterdir() if p.is_dir()}
    run(ROOT,[str(ROOT/'rebuild_lightweight.py'),str(work)],'compact_measurements')
    run(folders['06_mdlm'],['scripts/build_mdlm_paper_results.py'],'06_mdlm')
    run(ROOT,[str(ROOT/'rebuild_correction.py'),str(folders['09_reference_correction'])],'09_reference_correction')
    run(ROOT,[str(ROOT/'plot_correction.py'),'--source',str(folders['09_reference_correction']/'results'),
        '--output',str(OUT/'figures')],'09_figures')
    for script,component in [('rebuild_followup.py','10_correction_followup'),
                             ('rebuild_oracle_multiplier.py','11_oracle_multiplier'),
                             ('rebuild_bootstrap_confirmation.py','12_bootstrap_confirmation')]:
        run(ROOT,[str(ROOT/script),str(folders[component])],component)
    if '13_alpha_sweep' in folders:
        alpha=folders['13_alpha_sweep'];data=alpha/'results/alpha_sweep_v1';cfg=alpha/'configs/alpha_sweep_v1.json'
        for manifest in ['method_saved.json','analysis_saved.json']:
            for name,digest in read(data/manifest)['code_sha256'].items():
                actual=hashlib.sha256((alpha/name).read_bytes()).hexdigest()
                if actual!=digest:
                    assert manifest=='analysis_saved.json' and name in ['paper/make_alpha_sweep_figures.py','scripts/validate_alpha_sweep.py']
                    amendment=read(data/'presentation_amendment.json')['code_sha256'][name]
                    assert amendment==dict(before=digest,after=actual),name
                    assert hashlib.sha256((data/'analysis_code'/name).read_bytes()).hexdigest()==digest,name
        for name,digest in read(data/'forecasts_saved.json')['files'].items():
            assert hashlib.sha256((data/name).read_bytes()).hexdigest()==digest,name
        run(alpha,[str(alpha/'scripts/summarize_alpha_sweep.py'),'--source',str(data),'--config',str(cfg),
            '--output',str(data/'rebuilt')],'13_alpha_sweep')
        run(alpha,[str(alpha/'paper/make_alpha_sweep_figures.py'),'--source',str(data/'rebuilt'),'--config',str(cfg),
            '--output',str(OUT/'diagnostics/alpha_sweep')],'13_alpha_figures')
        run(ROOT,[str(ROOT/'rebuild_appendix_e.py'),str(work),'--output',str(OUT)],'appendix_e_and_table3')
    blocks=expected['tables'];verified={}
    def norm(line):return re.sub(r'\s+',' ',line).strip().rstrip('\\').strip()
    def rows(label,values,source):
        if label not in blocks:
            (OUT/'diagnostics'/(label.replace(':','_')+'.tex')).write_text('\n'.join(values)+'\n',encoding='utf-8');return
        actual={norm(line) for line in blocks[label].splitlines() if '&' in line}
        values=[line for line in values if line.strip()]
        for line in values:assert norm(line) in actual,(label,line)
        (OUT/'tables'/(label.replace(':','_')+'.tex')).write_text('\n'.join(values)+'\n',encoding='utf-8')
        verified[label]=dict(rows=len(values),source=source,mode='recomputed and reconciled',
                            table_number=expected['table_numbers'][label])
    rows('tab:occupancy',(folders['01_original_counts']/'tab_occupancy.tex').read_text().splitlines(),'01_original_counts: unordered queried-count histograms')
    neural=folders['06_mdlm']/'results/mdlm_paper_comparison_v1'
    for label,file in [('tab:mdlm-quality','quality_rows.tex'),('tab:mdlm-disagreement','disagreement_rows.tex'),('tab:mdlm-input-ranking','ranking_rows.tex')]:
        rows(label,(neural/file).read_text().splitlines(),'06_mdlm')
    summary=read_csv(neural/'summary.csv');values=[]
    for c,name in NAMES.items():
        cell=[next(r for r in summary if (r['corpus'],int(r['n']),r['method'],int(r['updates']))==(c,2048,method,updates)) for method,updates in [('lag',0),('mdlm',12000)]]
        vals=[f"{float(r['disagreement']):.4f}" for r in cell]+[f"{100*(1-float(r['mse'])/float(r['baseline_mse'])):.2f}" for r in cell]
        values.append(name+' & '+' & '.join(vals))
    rows('tab:main-mdlm',values,'06_mdlm/summary.csv')
    tables=read(folders['10_correction_followup']/'results/correction_followup_v1/rebuilt/tables.json')
    diagnostic=read(folders['11_oracle_multiplier']/'results/oracle_multiplier_v1/tables.json')
    for label,values in diagnostic.items():tables.setdefault(label,[]).extend(values)
    bootstrap=read(folders['12_bootstrap_confirmation']/'results/bootstrap_confirmation_v1/tables.json')
    tables.update(bootstrap)
    for label,values in tables.items():
        rows(label,values,'12_bootstrap_confirmation' if label in bootstrap else '10_correction_followup / 11_oracle_multiplier')
    if '13_alpha_sweep' in folders:
        for label,values in read(folders['13_alpha_sweep']/'results/alpha_sweep_v1/rebuilt/tables.json').items():
            rows(label,values,'13_alpha_sweep: paired bank and pair summaries')
        rows('tab:main-alpha-ablation',read(OUT/'APPENDIX_E_CHECKS.json')['table3_rows'],
             '13_alpha_sweep: source-split forecasts recomputed from full/half bank estimates, 20 pair outcomes, fixed weights, m<n')
    for label,mode in [('tab:image-text-map','conceptual correspondence'),('tab:data-roles','protocol overview')]:
        (OUT/'tables'/(label.replace(':','_')+'.tex')).write_text(blocks[label],encoding='utf-8')
        verified[label]=dict(mode=mode,source='Manuscript definitions. Not an empirical measurement.',
                            table_number=expected['table_numbers'][label])
    assert set(verified)==set(blocks),set(blocks)-set(verified)
    # Check completeness as well as agreement. Header rows are excluded using
    # the exact empirical row counts of the current manuscript inventory.
    required=expected['empirical_row_counts']
    for label,count in required.items():assert verified[label]['rows']==count,(label,verified[label]['rows'],count)
    sources={
        'fig_reference_correction.pdf':OUT/'figures/fig_reference_correction.pdf',
        'fig_reference_correction_all.pdf':OUT/'figures/fig_reference_correction_all.pdf',
        'fig_missing_context_decomposition.pdf':OUT/'figures/fig_missing_context_decomposition.pdf',
        'fig_mdlm_disagreement.pdf':neural/'mdlm_disagreement.pdf',
        'fig_multiplier_transfer.pdf':folders['11_oracle_multiplier']/'results/oracle_multiplier_v1/fig_multiplier_transfer.pdf',
        'fig_lag_topk_comparison.pdf':folders['05_topk']/'results/lag_topk_matched_v1/topk_comparison.pdf'}
    for name in list(sources):
        if name not in expected['figures']:
            source=sources.pop(name)
            shutil.copy2(source,OUT/'diagnostics'/name)
            if source.parent==OUT/'figures':source.unlink()
    assert set(sources)==set(expected['figures'])
    for name,source in sources.items():
        assert source.exists(),source
        if source.resolve()!=(OUT/'figures'/name).resolve():shutil.copy2(source,OUT/'figures'/name)
    shutil.copy2(neural/'mdlm_quality.pdf',OUT/'diagnostics/mdlm_quality.pdf')
    claims={}
    for c in NAMES:
        for method,updates in [('lag',0),('mdlm',12000)]:
            ds=[float(next(r['disagreement'] for r in summary if (r['corpus'],int(r['n']),r['method'],int(r['updates']))==(c,n,method,updates))) for n in (512,2048)]
            value=100*(1-ds[1]/ds[0]);assert f'{value:.1f}\\%' in expected['reported_percent_strings']
            claims[f'{c}_{method}_size_decline_percent']=value
    for r in read_csv(folders['07_directions']/'results/mdlm_subspaces_v1/summary.csv'):
        value=100*float(r['capture']);assert f'{value:.1f}\\%' in expected['reported_percent_strings']
        claims[f"{r['corpus']}_{r['method']}_projected_energy_percent"]=value
    logs=list((ROOT/'training_records').glob('*/*/training.jsonl'));assert len(logs)==66
    elapsed=sum(json.loads(p.read_text().strip().splitlines()[-1])['elapsed_seconds'] for p in logs)/3600
    assert f'{elapsed:.2f} hours' in expected['reported_hours'];claims['summed_training_hours']=elapsed
    (OUT/'PROSE_CLAIMS.json').write_text(json.dumps(claims,indent=2)+'\n')
    index=['# Reproduced submission outputs', '', 'Scope: main text and Appendices A through E.', '',
           '## Tables', '', '| Number | Output |', '|---|---|']
    for label,number in expected['table_numbers'].items():
        index.append(f'| {number} | [LaTeX rows](tables/{label.replace(":","_")}.tex) |')
    index+=['','Table 3 also has [CSV values](tables/table_3.csv) and [an independent calculation audit](APPENDIX_E_CHECKS.json).',
            '', '## Figures', '', '| Number | PDF |', '|---|---|']
    for name,number in expected['figure_numbers'].items():index.append(f'| {number} | [Figure {number}](figures/{name}) |')
    index+=['', 'Additional sweep plots and unsubmitted tables are in diagnostics/.',
            'They support the numerical record and are not additional submission appendices.']
    (OUT/'INDEX.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    empirical_count=sum(v['mode']=='recomputed and reconciled' for v in verified.values())
    report=dict(status='passed',empirical_tables=empirical_count,conceptual_tables=1,protocol_tables=1,figures=len(sources),
        tables=verified,seconds=time.perf_counter()-started,work_directory=str(work.relative_to(ROOT)),
        submission_scope=expected['scope'],table3_verified=True,appendix_e_checks='APPENDIX_E_CHECKS.json',
        scope='Offline numerical recomputation from compact saved measurements, exact synthetic population moments and recorded covariance blocks. No corpus tokens, new real-text forecasts, neural inference or training. Historical source audits are supplied as records, not rerun from raw inputs.')
    (OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {empirical_count} empirical tables, 1 conceptual table, 1 protocol table, {len(sources)} figures. See reproduced/REPORT.json.')

if __name__=='__main__':main()

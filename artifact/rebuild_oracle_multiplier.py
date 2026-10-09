"""Recompute new synthetic population moments and multiplier transfer tests."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-12)
    else:assert a==b,(a,b)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args()
    root=a.root;out=root/'results/oracle_multiplier_v1'
    expected={name:json.loads((out/name).read_text()) for name in
              ['oracle_summary.json','multiplier_summary.json']}
    sys.path.insert(0,str(root/'src'))
    for filename in ['test_population_oracle.py','test_multiplier_transfer.py','test_population_calibration.py']:
        spec=importlib.util.spec_from_file_location(filename[:-3],root/'tests'/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name in sorted(vars(module)):
            if name.startswith('test_'):getattr(module,name)()
    print('PASS exhaustive population and held-out-factor checks.',flush=True)
    for script in ['run_population_oracle.py','analyze_multiplier_transfer.py','build_oracle_multiplier_results.py']:
        subprocess.run([sys.executable,str(root/'scripts'/script)],cwd=root,check=True)
    for name,value in expected.items():compare(value,json.loads((out/name).read_text()))
    print('PASS exact population moments, multiplier transfers, intervals and figure.',flush=True)


if __name__=='__main__':main()

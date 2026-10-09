"""Check repository content, archive hashes, Python syntax, imports, and documentation links."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'artifact/PACKAGE_SHA256.json').read_text())
    for name, expected in manifest.items():
        path = ROOT/'artifact'/name
        assert path.is_file(), f'Missing evidence: {name}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Evidence changed: {name}'
    scripts = {p.stem for p in (ROOT/'scripts').glob('*.py')}
    python_files = [p for folder in ['src','scripts','tests','paper','runpod','colab']
                    for p in (ROOT/folder).rglob('*.py')]
    python_files.append(ROOT/'reproduce.py')
    for path in python_files:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('scripts.'):
                assert node.module.split('.')[1] in scripts, f'Missing local script import: {path}: {node.module}'
    docs = [ROOT/'README.md', ROOT/'CONTRIBUTING.md', *(ROOT/'docs').glob('*.md')]
    links = 0
    for path in docs:
        for command in re.findall(r'python (scripts/[^\s]+\.py)',path.read_text(encoding='utf-8')):
            assert (ROOT/command).is_file(), f'Broken experiment command: {command}'
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith(('#','mailto:')):
                continue
            target = unquote(target.split('#',1)[0].split(' ',1)[0]).strip('<>')
            assert (path.parent/target).exists(), f'Broken documentation link in {path.name}: {target}'
            links += 1
    inventory = json.loads((ROOT/'provenance/submitted_outputs.json').read_text())
    assert [t['number'] for t in inventory['tables']] == list(range(1,10))
    assert [f['number'] for f in inventory['figures']] == list(range(1,6))
    assert sum(t['kind']=='empirical' for t in inventory['tables']) == 7
    if (ROOT/'.git').exists():
        tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        prohibited = {'.pdf','.tex','.pt','.pth','.ckpt','.zip','.pem','.key','.sty','.bib'}
        assert all('artifact/'+name in tracked for name in manifest), 'Hashed evidence is missing from the Git index'
        assert not [p for p in tracked if Path(p).suffix.lower() in prohibited], 'Excluded files are tracked'
        assert not [p for p in tracked if p.startswith(('data/','local_data/','results/','artifact/reproduced/'))], 'Local data or generated outputs are tracked'
    print(json.dumps(dict(status='passed',evidence_hashes=len(manifest),python_files=len(python_files),
                          local_doc_links=links,submitted_tables=9,submitted_figures=5)))


if __name__ == '__main__':
    main()

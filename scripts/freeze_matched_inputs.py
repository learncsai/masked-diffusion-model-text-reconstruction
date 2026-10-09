"""Create the local input archive expected by the recorded count top-k evaluator."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'results/lag_topk_matched_v1/frozen_inputs.zip'
    if output.exists():
        raise FileExistsError('Frozen input archive already exists; use a fresh output working copy.')
    content = {}
    for corpus in ['tinystories','wikitext','cnn_dailymail']:
        folder = ROOT/'results/mdlm_runpod_main_v1'/corpus
        prepared = json.loads((folder/'prepared.json').read_text())
        for name in ['manifest.json','prepared.json','context.npz','lag_pairs.json','source_audit.json']:
            data = (folder/name).read_bytes()
            if name in prepared['sha256']:
                assert hashlib.sha256(data).hexdigest() == prepared['sha256'][name], name
            content[f'{corpus}/{name}'] = data
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in content.items():
            archive.writestr(name,data)
    print(output)


if __name__ == '__main__':
    main()

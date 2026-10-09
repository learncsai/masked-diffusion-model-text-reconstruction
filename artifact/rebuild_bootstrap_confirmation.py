"""Verify and rebuild the added separate-pair bootstrap comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from rebuild_oracle_multiplier import compare


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path)
    root = ap.parse_args().root
    out = root/'results/bootstrap_confirmation_v1'
    expected = json.loads((out/'summary.json').read_text())
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('bootstrap_checks', root/'tests/test_bootstrap_confirmation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.test_switch_preserves_banks_and_routes_equality_to_multinomial()
    subprocess.run([sys.executable, str(root/'scripts/summarize_bootstrap_confirmation.py')], cwd=root, check=True)
    compare(expected, json.loads((out/'summary.json').read_text()))
    subprocess.run([sys.executable, str(root/'scripts/validate_bootstrap_confirmation.py')], cwd=root, check=True)
    print('PASS separate-pair bootstrap estimates, source weights, paired intervals and switching rule.')


if __name__ == '__main__': main()

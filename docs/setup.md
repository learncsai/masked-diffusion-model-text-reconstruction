# Setup and reproduction

## Offline analysis environment

The original archive was validated with Python 3.14.2, NumPy 2.5.0, SciPy 1.16.3, and Matplotlib 3.10.8. Use Python 3.14 with `requirements.txt` for this path. Create and activate a virtual environment as shown in the [README](../README.md), then:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

The root command validates archive hashes, runs the numerical calculations, and exports the attached submission's nine tables and five figures to `results/reproduction/`. Seven tables are empirical calculations and two are protocol/concept definitions. The original archive's extra size-decision table is retained as a diagnostic.

The archive's own command remains available:

```bash
python artifact/reproduce.py
```

It rebuilds eight empirical tables plus two definition tables using the archive's older numbering. Prefer the root command and [current paper map](paper-map.md) for the submitted version. `python artifact/reproduce.py --extract-only` verifies hashes and copies evidence without running analyses.

## Full experiments environment

Use a separate **Python 3.12** environment for the recorded training stack. Install the CUDA-enabled PyTorch 2.8.0 build appropriate to your GPU first; the recorded experiment used CUDA 12.8. Install the remaining packages with:

```bash
python -m pip install -r requirements-experiments.txt
python -m pip install --no-deps -e .
```

The analysis package supports Python 3.12+, but offline replay uses Python 3.14. There is no need to install PyTorch for the offline path. New source acquisition additionally uses Hugging Face Datasets and Transformers; new count experiments run on CPU, while the full neural training recipe requires a CUDA GPU.

The original recorded environment is [runtime_environment.json](../artifact/training_records/runtime_environment.json): Python 3.12, PyTorch 2.8.0+cu128, NumPy 2.5.3, and SciPy 1.18.1. The shared analysis requirements use the versions tested for archive replay. The download dependencies use bounded version ranges because their historical exact versions were not recorded. Check `docs/validation.json` for the stack and checks exercised when this repository was assembled.

## Verification

```bash
python scripts/check_repository.py
python -m pytest tests/test_source_moment_forecast.py tests/test_reference_correction.py tests/test_population_oracle.py tests/test_alpha_sweep.py tests/test_public_preparation.py
```

These tests cover numerical identities, source-group resampling, finite-reference correction, exact synthetic moments, cached smoothing calculations, and source-disjoint preprocessing. Historical full-input integration audits need their recorded inputs; they are described in the archive and are not substitutes for fresh GPU validation.

GitHub Actions runs archive verification and complete numerical replay on CPU, and a separate job runs these scientific checks. It does not download corpora or allocate a cloud GPU.

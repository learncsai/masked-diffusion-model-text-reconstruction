# Troubleshooting

| Symptom | Likely cause and next action |
|---|---|
| `No matching distribution` installing offline pins | Use Python 3.14 and a current pip. Keep the Python 3.12 GPU environment separate. |
| `No module named diffusion_lm_rmt` | Activate the experiment environment, run `python -m pip install --no-deps -e .`, and launch commands from the repository root. |
| Archive hash assertion | A tracked evidence file changed, or line endings were altered. Restore from a clean clone; preserve `artifact/** -text` Git attributes. |
| `... failed. Read reproduced/logs/...` | Open the corresponding `artifact/reproduced/logs/` file. The last calculation failure is recorded there. |
| Public preparation says inputs already exist | Use a fresh working copy for a different preparation; the command does not overwrite partially created or frozen scientific inputs. |
| Historical script asks for an omitted `.npz`, `.pt`, or manifest | It expects a historical input export. Use the fresh preparation recipe for a new matched/correction replication; do not bypass input checks. |
| Dataset or tokenizer hash mismatch during recovery | Acquired text/order/tokenization differs. Choose the recorded revision if available; otherwise report a changed-input replication. |
| Frozen code/design signature changed | Use a new output label or working copy. Do not edit saved signatures to suppress the check. |
| Correction initialization fails development gate | Run the full synthetic development configuration and inspect every case. A smoke pilot is not the full gate. |
| `CUDA GPU is required` | Run offline reproduction on CPU, or run the full training recipe on an existing CUDA machine. |
| Neural evaluator cannot find `mdlm_runpod_main_v1` | Train with the default label and complete all corpora, or adapt every embedded run path consistently. |
| Neural input/context hashes differ | Training and scoring must use the same saved masks, auxiliary quantities, architecture, and token arrays. Restore the matching input set. |

Generated `results/` and `artifact/reproduced/` can be removed locally to reclaim space. Each archive replay preserves a separate timestamped working copy. The tracked compact evidence remains in `artifact/components/`.

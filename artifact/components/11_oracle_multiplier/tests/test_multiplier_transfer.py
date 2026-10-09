import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/analyze_multiplier_transfer.py'
spec=importlib.util.spec_from_file_location('transfer',path)
transfer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def test_heldout_corpus_and_cells_cannot_change_selected_factor():
    data=np.random.default_rng(12).normal(size=(3,6,3,4,4))
    cells=[(0,0),(1,1),(2,2)]
    baseline=transfer.train_factor(data,cells,[0,1])
    changed=data.copy()
    changed[2]+=100
    changed[:,:,2,3]+=100
    np.testing.assert_equal(transfer.train_factor(changed,cells,[0,1]),baseline)
    # Independent explicit list of the intended training records.
    values=[data[c,b,m,n,2] for c in [0,1] for b in range(6) for m,n in cells]
    assert baseline == -np.median(values)
    batched=np.stack([data,changed])
    np.testing.assert_equal(transfer.train_factor(batched,cells,[0,1]),[baseline,baseline])


def test_error_uses_mean_of_bank_errors_not_error_of_mean_forecast():
    logs=np.zeros((3,2,1,1,4))
    logs[:,0,0,0,2]=-1
    logs[:,1,0,0,2]=1
    assert transfer.score(logs,[(0,0)],0).item()==1
    np.testing.assert_equal(transfer.score(np.stack([logs,logs]),[(0,0)],np.array([0.,0.])),[[1.],[1.]])

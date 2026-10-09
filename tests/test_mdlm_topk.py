"""Check matched neural ranking and scoring, including special-token exclusion."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_mdlm_topk import score_checkpoint


class FixedDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Artificial MASK/PAD outputs deliberately dominate every content logit.
        self.register_buffer("scores", torch.tensor([
            [[0., 0., 0., 100., 200.]]*3,
            [[0., 2., 1., 100., 200.], [1., 0., 0., 100., 200.], [1., 1., 0., 100., 200.]],
        ]))
        self.seen = None

    def forward(self, ids, attention, times):
        self.seen = (ids.clone(), attention.clone(), times.clone())
        return self.scores


def test_neural_topk_masks_targets_excludes_specials_and_keeps_eos_and_ties():
    model = FixedDenoiser()
    ids = np.array([[2, 0, 1], [1, 2, 0]])  # Content ID 2 stands for EOS.
    mask = np.array([[True, False, False], [True, False, True]])
    config = dict(classes=3, mask_token_id=3, mask_rate=.5, training=dict(microbatch_size=2))
    result = score_checkpoint(model, ids, mask, config, torch.device("cpu"))
    np.testing.assert_array_equal(result["target_rank"], [3, 1, 1])
    np.testing.assert_array_equal(result["best_tie_rank"], [1, 1, 1])
    np.testing.assert_array_equal(result["worst_tie_rank"], [3, 1, 2])
    np.testing.assert_array_equal(model.seen[0].numpy(), np.where(mask, 3, ids))
    assert model.seen[1].all() and (model.seen[2] == .5).all()
    p = torch.softmax(model.scores[..., :3], -1)
    one_hot = torch.nn.functional.one_hot(torch.as_tensor(ids), 3)
    squared = (p-one_hot).square().sum(-1)
    expected = np.array([squared[i][mask[i]].double().mean().item() for i in range(2)])
    np.testing.assert_allclose(result["mse_by_chunk"], expected, atol=1e-7)
    assert result["mse_by_chunk"].mean() != pytest.approx(squared[mask].mean().item())


def test_empty_chunk_masks_are_rejected():
    config = dict(classes=3, mask_token_id=3, mask_rate=.5, training=dict(microbatch_size=2))
    with pytest.raises(ValueError, match="masks"):
        score_checkpoint(FixedDenoiser(), np.zeros((2, 3), dtype=int),
                         np.zeros((2, 3), dtype=bool), config, torch.device("cpu"))

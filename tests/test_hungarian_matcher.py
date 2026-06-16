import unittest

import torch
from scipy.optimize import linear_sum_assignment

from heptv2.training.losses import (
    HungarianMatcher,
    _apply_point_order,
    _maybe_get_point_order,
    batch_dice_loss,
    batch_sigmoid_focal_loss,
)


def _legacy_assignment(outputs, targets):
    """Reproduce the pre-fix matcher target flattening bug."""
    out_prob = outputs["pred_logits"][0].softmax(-1)
    out_mask = outputs["pred_masks"][0]
    tgt_ids = targets[0]["labels"]
    tgt_mask = targets[0]["masks"].to(out_mask)
    tgt_mask = _apply_point_order(
        tgt_mask,
        _maybe_get_point_order(outputs, batch_idx=0, device=tgt_mask.device),
    )

    cost_class = -out_prob[:, tgt_ids]
    out_flat = out_mask.flatten(1)
    legacy_tgt_flat = tgt_mask[:, 0].flatten(1)
    cost_mask = batch_sigmoid_focal_loss(out_flat, legacy_tgt_flat)
    cost_dice = batch_dice_loss(out_flat, legacy_tgt_flat)
    cost = cost_class * 0.0 + cost_mask + cost_dice
    return linear_sum_assignment(cost.float().cpu())


def _make_cross_match_case(point_order=None):
    target0 = torch.tensor([0.0, 1.0, 1.0, 0.0])
    target1 = torch.tensor([0.0, 0.0, 0.0, 1.0])
    targets = [
        {
            "labels": torch.zeros(2, dtype=torch.long),
            "masks": torch.stack([target0, target1]).unsqueeze(-1),
        }
    ]
    if point_order is None:
        point_order = torch.arange(4)
    serialized_targets = targets[0]["masks"].index_select(1, point_order)
    query0_matches_target1 = serialized_targets[1].mul(20.0).sub(10.0)
    query1_matches_target0 = serialized_targets[0].mul(20.0).sub(10.0)
    outputs = {
        "pred_logits": torch.zeros(1, 2, 2),
        "pred_masks": torch.stack([query0_matches_target1, query1_matches_target0]).unsqueeze(0),
        "_serialized_point_order": point_order,
    }
    return outputs, targets


class HungarianMatcherTest(unittest.TestCase):
    def test_legacy_matcher_target_flatten_drops_all_but_first_hit(self):
        outputs, targets = _make_cross_match_case()

        tgt_mask = targets[0]["masks"]
        legacy_tgt_flat = tgt_mask[:, 0].flatten(1)
        correct_tgt_flat = tgt_mask.flatten(1)

        self.assertEqual(legacy_tgt_flat.shape, (2, 1))
        self.assertEqual(correct_tgt_flat.shape, (2, 4))
        self.assertTrue(torch.equal(legacy_tgt_flat, torch.zeros(2, 1)))

        legacy_src, legacy_tgt = _legacy_assignment(outputs, targets)
        self.assertEqual(legacy_src.tolist(), [0, 1])
        self.assertEqual(legacy_tgt.tolist(), [0, 1])
        self.assertNotEqual(legacy_tgt.tolist(), [1, 0])

    def test_hungarian_matcher_uses_full_target_masks(self):
        outputs, targets = _make_cross_match_case()

        matcher = HungarianMatcher(cost_class=0.0, cost_mask=1.0, cost_dice=1.0)
        src_idx, tgt_idx = matcher(outputs, targets)[0]

        self.assertEqual(src_idx.tolist(), [0, 1])
        self.assertEqual(tgt_idx.tolist(), [1, 0])

    def test_hungarian_matcher_reorders_targets_to_serialized_point_order(self):
        outputs, targets = _make_cross_match_case(point_order=torch.tensor([2, 0, 3, 1]))

        matcher = HungarianMatcher(cost_class=0.0, cost_mask=1.0, cost_dice=1.0)
        src_idx, tgt_idx = matcher(outputs, targets)[0]

        self.assertEqual(src_idx.tolist(), [0, 1])
        self.assertEqual(tgt_idx.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()

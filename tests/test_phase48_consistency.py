import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
INFER_CONFIG = REPO_ROOT / "heptv2" / "configs" / "finetune_phase48_scatter_amp.yaml"
TRAIN_CONFIG = REPO_ROOT / "heptv2" / "configs" / "finetune_phase48_scatter_amp_train.yaml"
ENV_HELPER = REPO_ROOT / "codex_tools" / "env_phase48_scatter_amp_best.sh"


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


class Phase48ConfigConsistencyTest(unittest.TestCase):
    def test_train_config_matches_best_inference_path(self):
        infer_cfg = _load_yaml(INFER_CONFIG)
        train_cfg = _load_yaml(TRAIN_CONFIG)

        self.assertEqual(train_cfg["resume"], infer_cfg["checkpoint_path"])
        for section in ("model_kwargs", "inference_path", "postprocessing"):
            self.assertEqual(train_cfg[section], infer_cfg[section], section)
        for key in ("enabled", "dtype"):
            self.assertEqual(train_cfg["amp"][key], infer_cfg["amp"][key], f"amp.{key}")
        self.assertTrue(train_cfg["amp"]["grad_scaler"])
        self.assertEqual(train_cfg["amp"]["init_scale"], 1)

        for key in (
            "eta_abs_max",
            "pt_thld",
            "predicted_count_thld",
            "min_track_length",
            "num_sub_events",
        ):
            self.assertEqual(train_cfg["eval"][key], infer_cfg["eval"][key], f"eval.{key}")

        self.assertEqual(train_cfg["eval"]["metric_path"], "inference")
        self.assertFalse(train_cfg["eval"]["compute_loss"])
        self.assertEqual(train_cfg["best_metric_key"], "dm")
        self.assertEqual(train_cfg["best_metric_mode"], "max")
        self.assertEqual(train_cfg["env_file"], str(ENV_HELPER.relative_to(REPO_ROOT)))
        self.assertEqual(train_cfg["model_kwargs"]["decoder_serialization_type"], "none")
        self.assertFalse(train_cfg["model_kwargs"]["decoder_overlap"])

    def test_none_serialized_point_order_uses_valid_mask_not_prefix_trim(self):
        from heptv2.training.train import trim_pred_by_valid_mask

        pred_masks = torch.arange(1 * 2 * 5, dtype=torch.float32).reshape(1, 2, 5)
        aux_masks = (pred_masks + 100).clone()
        pred = {
            "pred_masks": pred_masks.clone(),
            "_serialized_point_order": None,
            "aux_outputs": [
                {
                    "pred_masks": aux_masks.clone(),
                    "_serialized_point_order": None,
                }
            ],
        }
        valid_mask = torch.tensor([True, False, True, False, True])

        out = trim_pred_by_valid_mask(pred, valid_mask)

        self.assertIsNone(out["_serialized_point_order"])
        self.assertIsNone(out["aux_outputs"][0]["_serialized_point_order"])
        torch.testing.assert_close(out["pred_masks"], pred_masks[:, :, valid_mask])
        torch.testing.assert_close(out["aux_outputs"][0]["pred_masks"], aux_masks[:, :, valid_mask])
        self.assertFalse(torch.equal(out["pred_masks"], pred_masks[:, :, : int(valid_mask.sum())]))

    def test_run_train_applies_env_before_postprocess_import(self):
        code = f"""
import sys
import heptv2.run_train
assert 'heptv2.eval.postprocess' not in sys.modules
from heptv2.run_train import _apply_env_file
_apply_env_file({str(ENV_HELPER)!r})
import heptv2.eval.postprocess as pp
assert pp._CODEX_ASSIGN_FP16 is True
assert pp._CODEX_DO_FDM_TREE_DROP is True
assert pp._CODEX_DO_FDM2_TREE_DROP is True
assert pp._CODEX_DEDUP_MIN_SIZE_AFTER == 2
assert abs(pp._CODEX_REFINE_MARGIN - 0.003) < 1e-12
"""
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )


@unittest.skipUnless(
    os.environ.get("HEPTV2_RUN_GPU_CONSISTENCY_TESTS") == "1",
    "set HEPTV2_RUN_GPU_CONSISTENCY_TESTS=1 to run the one-event GPU consistency test",
)
class Phase48GpuMetricConsistencyTest(unittest.TestCase):
    def test_train_valid_test_metric_epoch_matches_run_inference_event_on_one_event_each(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        from torch_geometric.loader import DataLoader

        from heptv2.data.dataset import TrackingTransform, TrackmlLarge
        from heptv2.model import Transformer
        from heptv2.run_train import _apply_env_file, _run_inference_metrics_epoch
        from heptv2.training.train_utils import setup_amp

        cfg = _load_yaml(TRAIN_CONFIG)
        cfg["device"] = os.environ.get("HEPTV2_TEST_DEVICE", cfg["device"])
        device = torch.device(cfg["device"])
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")

        checkpoint = REPO_ROOT / cfg["resume"]
        if not checkpoint.exists():
            self.skipTest(f"checkpoint not found: {checkpoint}")

        data_root = REPO_ROOT / cfg["data_root"]
        if not data_root.exists():
            self.skipTest(f"data root not found: {data_root}")

        _apply_env_file(REPO_ROOT / cfg["env_file"])
        from heptv2.run_inference import _load_checkpoint, run_event

        dataset = TrackmlLarge(root=data_root, transform=TrackingTransform())
        model = Transformer(
            attn_type="hept",
            in_dim=dataset.x_dim,
            coords_dim=dataset.coords_dim,
            task=cfg["dataset_name"],
            **cfg["model_kwargs"],
        ).to(device)
        model.eval()
        _load_checkpoint(model, checkpoint, device)

        amp_enabled, amp_dtype, _ = setup_amp(cfg, device)
        for split in ("train", "valid", "test"):
            with self.subTest(split=split):
                indices = dataset.idx_split[split]
                if len(indices) == 0:
                    self.skipTest(f"{split} split is empty")
                event = dataset[int(indices[0])]
                loader = DataLoader([event], batch_size=1, shuffle=False, num_workers=0)
                data = next(iter(DataLoader([event], batch_size=1, shuffle=False, num_workers=0)))
                direct_metrics = run_event(model, data, cfg, device, amp_enabled, amp_dtype)
                epoch_metrics = _run_inference_metrics_epoch(
                    model,
                    loader,
                    split,
                    0,
                    device,
                    cfg,
                    amp_enabled,
                    amp_dtype,
                    limit_batches=1,
                )

                for key in ("dm", "fake_double_majority", "technical_efficiency", "fake_rate"):
                    self.assertIn(key, direct_metrics)
                    self.assertIn(key, epoch_metrics)
                    self.assertAlmostEqual(float(direct_metrics[key]), float(epoch_metrics[key]), places=10)


if __name__ == "__main__":
    unittest.main()

"""Offline crop embedding boundary; no detector checkpoint is needed."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessor

from rank_homes.embeddings import ARTIFACT_FILES, DinoV3Embedder, record_artifacts


class EmbeddingTests(unittest.TestCase):
    def stage(self, root):
        for name in ARTIFACT_FILES:
            (root / name).write_text("fixture only", encoding="utf-8")
        DINOv3ViTImageProcessor().save_pretrained(root)
        record_artifacts(root, "a" * 40)

    def test_invalid_cls_and_incomplete_weights_are_rejected(self):
        class Model(torch.nn.Module):
            config = DINOv3ViTConfig(num_register_tokens=4)
            output = torch.zeros(1, 384)

            def forward(self, **kwargs):
                return SimpleNamespace(pooler_output=self.output)

        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.stage(root)
            model = Model()
            with patch("transformers.DINOv3ViTModel.from_pretrained", return_value=(model, {})):
                embedder = DinoV3Embedder(root, device="cpu")
            with Image.new("RGB", (8, 4)) as crop:
                for output in (torch.zeros(1, 384), torch.full((1, 384), 1e-15),
                               torch.full((1, 384), float("nan")), torch.ones(1, 768)):
                    model.output = output
                    with self.assertRaisesRegex(ValueError, "Invalid DINOv3 CLS"):
                        embedder(crop)
            with patch("transformers.DINOv3ViTModel.from_pretrained",
                       return_value=(model, {"missing_keys": ["cls_token"]})):
                with self.assertRaisesRegex(ValueError, "Incomplete or incompatible"):
                    DinoV3Embedder(root)

    def test_changed_artifacts_are_rejected_before_model_loading(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.stage(root)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch: model.safetensors"):
                DinoV3Embedder(root)

    def test_crop_is_rescaled_once_and_normalized_cls_is_returned(self):
        class Model(torch.nn.Module):
            config = DINOv3ViTConfig(num_register_tokens=4)

            def forward(self, pixel_values):
                self_test.assertFalse(self.training)
                self_test.assertTrue(torch.is_inference_mode_enabled())
                self_test.assertEqual(pixel_values.dtype, torch.float32)
                self_test.assertEqual(tuple(pixel_values.shape), (1, 3, 224, 224))
                # Entire elongated image survives resize; no center crop or padding.
                np.testing.assert_allclose(pixel_values[0, :, 0, 0],
                    [-2.117904, -2.035714, -1.804444], atol=1e-5)
                np.testing.assert_allclose(pixel_values[0, :, -1, -1],
                    [2.248908, 2.428571, 2.64], atol=1e-5)
                from types import SimpleNamespace
                return SimpleNamespace(pooler_output=torch.tensor([[3., 4.] + [0.] * 382]))

        self_test = self
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ARTIFACT_FILES:
                (root / name).write_text("fixture only", encoding="utf-8")
            DINOv3ViTImageProcessor().save_pretrained(root)
            record_artifacts(root, "a" * 40)
            with patch("transformers.DINOv3ViTModel.from_pretrained", return_value=(Model(), {})):
                embedder = DinoV3Embedder(root, device="cpu")
            pixels = np.zeros((10, 100, 3), dtype=np.uint8)
            pixels[:, 50:] = 255
            with Image.fromarray(pixels) as crop:
                vector = embedder(crop)
            self.assertEqual((vector.dtype, vector.shape), (np.dtype("float32"), (384,)))
            np.testing.assert_allclose(vector[:2], [0.6, 0.8], atol=1e-7)

    def test_missing_artifacts_fail_without_contacting_hub(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Incomplete local DINOv3 artifacts"):
                DinoV3Embedder(Path(folder))


if __name__ == "__main__":
    unittest.main()

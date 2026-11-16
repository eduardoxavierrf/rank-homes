"""Opt-in real-crop validation, never a search quality benchmark."""
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from rank_homes.domain import Photo
from rank_homes.embeddings import DinoV3Embedder
from rank_homes.vision import Detection, GeometryExtractor


@unittest.skipUnless(os.environ.get("DINOV3_ARTIFACTS") and os.environ.get("DINOV3_CROP"),
                     "approved local artifacts and a real object crop are required")
class RealCropSmokeTests(unittest.TestCase):
    def test_real_crop_repeated_inference_and_alignment(self):
        path = Path(os.environ["DINOV3_CROP"])
        embedder = DinoV3Embedder(Path(os.environ["DINOV3_ARTIFACTS"]))
        digest = sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source, source.convert("RGB") as crop:
            photo = Photo("smoke-crop", "reference", "smoke", "kitchen", path.name,
                          digest, crop.width, crop.height)
            box = (0, 0, crop.width, crop.height)
            extractor = GeometryExtractor(lambda image: [Detection(0, 1., box)], embedder,
                classes={0: "sink"}, dimension=384, configuration=embedder.configuration)
            first = extractor.analyze(photo, crop)
            second = extractor.analyze(photo, crop)
        self.assertEqual(first.vectors.shape, (1, 384))
        self.assertEqual(first.vectors.dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(first.vectors).all())
        self.assertEqual(first.regions, second.regions)
        self.assertEqual(first.regions[0].box, box)
        np.testing.assert_allclose(np.linalg.norm(first.vectors, axis=1), [1.], atol=1e-5)
        similarity = float(first.vectors[0] @ first.vectors[0])
        self.assertAlmostEqual(similarity, 1., places=5)
        np.testing.assert_allclose(first.vectors, second.vectors, atol=1e-6, rtol=1e-6)
        print(json.dumps({"validation_only": True, "crop_sha256": digest,
            "region_id": first.regions[0].region_id, "box": box,
            "shape": list(first.vectors.shape), "finite": True,
            "norms": np.linalg.norm(first.vectors, axis=1).tolist(),
            "self_similarity": similarity,
            "repeat_max_abs_difference": float(np.abs(first.vectors - second.vectors).max()),
            "configuration": embedder.configuration}, indent=2))

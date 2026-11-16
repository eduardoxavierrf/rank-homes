"""Opt-in real offline search evidence; source catalogs stay untouched."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import socket
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from rank_homes.catalog import load_catalog, read_image
from rank_homes.cli import main
from rank_homes.vision import VisualExtractor, create_extractor


@unittest.skipUnless(os.environ.get("RANK_HOMES_REAL_SMOKE"), "requires staged models and real local catalogs")
class OfflineSearchSmokeTests(unittest.TestCase):
    def test_real_extraction_and_commands_with_network_disabled(self):
        references = Path("data/references")
        listings = Path("data/candidates")
        before = {str(p): p.read_bytes() for root in (references, listings) for p in root.rglob("*") if p.is_file()}
        # Deny socket connections, including downloads attempted by dependency code.
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network disabled")), \
                patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network disabled")):
            extractor = create_extractor()
            import torch
            self.assertFalse(extractor.model.model.model.training)
            self.assertFalse(extractor.embedder.model.training)
            for model in (extractor.model.model.model, extractor.embedder.model):
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            metrics = []
            for room in ("kitchen", "bathroom"):
                photo = next(p for p in load_catalog(references, "reference") if p.room == room)
                with read_image(references / photo.relative_path) as image:
                    first = extractor.analyze(photo, image)
                    second = extractor.analyze(photo, image)
                    self.assertGreater(len(first.regions), 0)
                    self.assertEqual(first.regions, second.regions)
                    self.assertEqual(first.vectors.shape, (len(first.regions), 384))
                    self.assertEqual(first.vectors.dtype, np.dtype("float32"))
                    self.assertTrue(np.isfinite(first.vectors).all())
                    np.testing.assert_allclose(np.linalg.norm(first.vectors, axis=1), 1, atol=1e-5)
                    np.testing.assert_allclose(first.vectors, second.vectors, atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(np.sum(first.vectors * first.vectors, axis=1), 1, atol=1e-5)
                    if room == "kitchen":
                        # The public cutoff contract includes confidence equality.
                        boundary = max(r.confidence for r in first.regions)
                        inclusive = create_extractor(cutoff=boundary)
                        at_cutoff = inclusive.analyze(photo, image)
                        self.assertTrue(any(r.confidence == boundary for r in at_cutoff.regions))
                        del inclusive
                    for region in first.regions:
                        with image.crop(region.box) as crop:
                            self.assertGreater(min(crop.size), 0)
                    metrics.append({"room": room, "source": photo.relative_path, "size": image.size,
                        "regions": [{"class": r.object_class, "box": r.box, "confidence": r.confidence} for r in first.regions],
                        "shape": first.vectors.shape, "norms": np.linalg.norm(first.vectors, axis=1).tolist(),
                        "repeat_max_abs_difference": float(np.abs(first.vectors-second.vectors).max())})
            with TemporaryDirectory(prefix="rank-homes-real-") as temporary:
                library = Path(temporary) / "library"
                def command(arguments):
                    output, errors = StringIO(), StringIO()
                    with redirect_stdout(output), redirect_stderr(errors):
                        status = main(arguments)
                    self.assertEqual(status, 0, errors.getvalue())
                    print(output.getvalue())
                    return output.getvalue()
                command(["references", "build", "--images", str(references), "--output", str(library)])
                published = {str(p): p.read_bytes() for p in library.rglob("*") if p.is_file()}
                for threshold in ("0", "0.7"):
                    output = command(["compare", "--references", str(library), "--references-images", str(references),
                                      "--listings", str(listings), "--threshold", threshold])
                    self.assertIn("Evaluated:", output)
                    if threshold == "0":
                        for value in ("Room kitchen:", "Room bathroom:", "Candidate:", "Reference:", "oriented size", "box ("):
                            self.assertIn(value, output)
                self.assertEqual(published, {str(p): p.read_bytes() for p in library.rglob("*") if p.is_file()})
                moved = Path(temporary) / "moved-references"
                shutil.copytree(references, moved)
                source = next(p for p in moved.rglob("*") if p.is_file())
                source.write_bytes(source.read_bytes() + b"inventory-change")
                for extra, images in [(["--detector-confidence", "0.6"], references), ([], moved)]:
                    output, errors = StringIO(), StringIO()
                    with patch.object(VisualExtractor, "analyze", side_effect=AssertionError("candidate inference before validation")), \
                            redirect_stdout(output), redirect_stderr(errors):
                        status = main(["compare", "--references", str(library), "--references-images", str(images),
                                       "--listings", str(listings), *extra])
                    self.assertEqual(status, 1, errors.getvalue())
                    self.assertIn("rebuild", errors.getvalue().lower())
                    self.assertNotIn("Listing:", output.getvalue())
                    print("Expected stale-library rejection:", errors.getvalue())
            self.assertEqual(before, {str(p): p.read_bytes() for root in (references, listings) for p in root.rglob("*") if p.is_file()})
            print(json.dumps({"configuration": extractor.configuration, "metrics": metrics}, indent=2, default=str))

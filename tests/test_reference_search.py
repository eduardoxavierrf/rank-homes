"""The production command seam, with only inference replaced by a fixture."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from rank_homes.cli import main
from rank_homes.domain import Analysis, Region


class FixtureExtractor:
    dimension = 2
    configuration = {
        "extractor_version": "test-fixture-v1", "dimension": 2, "dtype": "float32",
        "detector_weights": "fixture-detector", "detector_weights_sha256": "a" * 64,
        "classes": {0: "sink"}, "detector_cutoff": 0.5,
        "geometry": "exif-rgb-clamp-floor-ceil-exclusive-v1",
        "embedding_weights": "fixture-embedding", "embedding_weights_sha256": "b" * 64,
        "embedding_revision": "fixture-v1", "processor": {"resize": [224, 224], "interpolation": "bilinear",
            "rescale_factor": 1 / 255, "mean": [0.5] * 3, "std": [0.5] * 3, "center_crop": False},
        "pooling": "CLS", "normalization": "L2",
    }

    def analyze(self, photo, image):
        # A known cosine of approximately 0.8 against the reference axis.
        vector = [1.0, 0.0] if photo.side == "reference" else [0.8, 0.6]
        region = Region(f"{photo.photo_id}:sink", photo, "sink", 0.9, (1, 2, 7, 6))
        return Analysis((region,), np.array([vector], dtype=np.float32))


class ReferenceSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="rank-homes-search-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.references = self.root / "references"
        self.listings = self.root / "listings"
        self.library = self.root / "library"
        for root, apartment in [(self.references, "desired"), (self.listings, "rental-001")]:
            folder = root / apartment / "kitchen"
            folder.mkdir(parents=True)
            Image.new("RGB", (10, 8), (40, 80, 120)).save(folder / "photo.png")

    def command(self, arguments, factory=FixtureExtractor):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(arguments, extractor_factory=factory)
        return status, output.getvalue(), errors.getvalue()

    def build(self, factory=FixtureExtractor):
        return self.command([
            "references", "build", "--images", str(self.references), "--output", str(self.library),
        ], factory)

    def compare(self, threshold="0.70", factory=FixtureExtractor):
        return self.command([
            "compare", "--references", str(self.library),
            "--references-images", str(self.references), "--listings", str(self.listings),
            "--threshold", threshold,
        ], factory)

    def test_build_and_compare_print_score_and_original_source_evidence(self):
        sources_before = {path: path.read_bytes() for path in self.root.rglob("*.png")}
        status, output, errors = self.build()
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Published reference library", output)

        status, output, errors = self.compare()
        self.assertEqual((status, errors), (0, ""))
        for expected in [
            "rental-001", str(self.listings / "rental-001"), "Score: 0.8000", "Threshold: 0.7000",
            "kitchen", "sink", "(1, 2, 7, 6)", "Matching photo:",
            str(self.listings / "rental-001/kitchen/photo.png"),
            str(self.references / "desired/kitchen/photo.png"),
            "Evaluated: 1; Selected: 1; Below threshold: 0; Skipped: 0",
        ]:
            self.assertIn(expected, output)
        self.assertEqual({path: path.read_bytes() for path in self.listings.rglob("*") if path.is_file()},
                         {path: data for path, data in sources_before.items() if path.is_relative_to(self.listings)})
        self.assertTrue(all(path.read_bytes() == data for path, data in sources_before.items()))

        pointer = json.loads((self.library / "CURRENT.json").read_text())
        generation = self.library / "generations" / pointer["generation"]
        manifest = json.loads((generation / "manifest.json").read_text())
        self.assertEqual(manifest["format_version"], 1)
        matrix = np.load(generation / "embeddings.npy", allow_pickle=False)
        self.assertEqual((matrix.dtype, matrix.shape), (np.dtype("float32"), (1, 2)))
        region = manifest["regions"][0]
        self.assertEqual(region["row"], 0)
        self.assertEqual(region["photo"]["side"], "reference")
        self.assertEqual(region["photo"]["apartment"], "desired")
        self.assertEqual(region["photo"]["relative_path"], "desired/kitchen/photo.png")
        self.assertEqual((region["photo"]["width"], region["photo"]["height"]), (10, 8))
        self.assertEqual(len(region["photo"]["sha256"]), 64)
        with Image.open(generation / region["crop_path"]) as crop:
            self.assertEqual((crop.format, crop.size, crop.getpixel((0, 0))), ("PNG", (6, 4), (40, 80, 120)))

    def test_selection_uses_unrounded_score_and_excludes_exact_threshold(self):
        self.assertEqual(self.build()[0], 0)
        status, output, errors = self.compare()
        raw_score = output.split("Comparison value: ")[1].split(" > ")[0]
        status, output, errors = self.compare(raw_score)
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Selected: 0; Below threshold: 1", output)
        below = str(np.nextafter(float(raw_score), -np.inf))
        status, output, errors = self.compare(below)
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Score: 0.8000; Threshold: 0.8000", output)
        self.assertIn("Selected: 1", output)

    def test_production_invocation_fails_clearly_without_real_extractor(self):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(["references", "build", "--images", str(self.references), "--output", str(self.library)])
        self.assertEqual(status, 1)
        self.assertIn("Real visual extraction is not integrated", errors.getvalue())
        self.assertFalse(self.library.exists())

    def test_import_help_and_catalog_do_not_import_models(self):
        import subprocess
        import sys
        code = """
import sys
import rank_homes
from pathlib import Path
from rank_homes.catalog import load_catalog
from rank_homes.cli import main
load_catalog(Path(sys.argv[1]), 'reference')
try:
    main(['--help'], extractor_factory=lambda: (_ for _ in ()).throw(AssertionError('loaded extractor')))
except SystemExit as error:
    assert error.code == 0
assert not {'torch', 'rfdetr', 'transformers'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.references)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

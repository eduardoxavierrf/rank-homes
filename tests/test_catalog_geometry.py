"""Catalog and geometry contracts through production build/compare commands."""
from hashlib import sha256
import json
import shutil
import unittest
import warnings

import numpy as np
from PIL import Image

import test_reference_search as search
from rank_homes.vision import Detection, GeometryExtractor


class CatalogGeometryTests(unittest.TestCase):
    setUp = search.ReferenceSearchTests.setUp
    command = search.ReferenceSearchTests.command
    build = search.ReferenceSearchTests.build
    compare = search.ReferenceSearchTests.compare

    def factory(self, detections=None, cutoff=0.5):
        detections = detections if detections is not None else [Detection(0, 0.5, (-0.2, 1.2, 12.2, 7.1))]
        return lambda: GeometryExtractor(lambda image: detections,
            lambda crop: np.array([1, 0], dtype=np.float32),
            classes={0: "sink"}, dimension=2, configuration={"fixture": "geometry"}, cutoff=cutoff)

    def manifest(self):
        pointer = json.loads((self.library / "CURRENT.json").read_text())
        folder = self.library / "generations" / pointer["generation"]
        return json.loads((folder / "manifest.json").read_text()), folder

    def test_inventory_keeps_duplicates_but_regions_use_group_representatives(self):
        original = self.references / "desired/kitchen/photo.png"
        shutil.copyfile(original, original.with_name("A.png"))
        for group in ["desired/bathroom", "other/kitchen"]:
            folder = self.references / group
            folder.mkdir(parents=True)
            shutil.copyfile(original, folder / "photo.png")
        for relative in ["outside.png", "desired/bedroom/photo.png", "desired/kitchen/deeper/photo.png"]:
            path = self.references / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(self.build(self.factory())[0], 0)
        manifest, _ = self.manifest()
        self.assertEqual(len(manifest["inventory"]), 4)
        self.assertEqual([r["photo"]["relative_path"] for r in manifest["regions"]],
                         ["desired/bathroom/photo.png", "desired/kitchen/A.png", "other/kitchen/photo.png"])
        messages = "\n".join(str(w.message) for w in caught)
        self.assertEqual(messages.count("outside apartment/room layout"), 3)
        self.assertIn("duplicate of desired/kitchen/A.png", messages)
        original.with_name("A.png").unlink()
        self.assertIn("inventory changed", self.compare(factory=self.factory())[2])

    def test_fractional_boundary_one_pixel_and_invalid_detections(self):
        detections = [Detection(0, 0.5, (-1.2, 1.2, 12.2, 7.1)),
                      Detection(0, 0.8, (2.1, 3.1, 2.2, 3.2)),
                      Detection(0, 0.49, (0, 0, 1, 1)),
                      Detection(0, float("nan"), (0, 0, 1, 1)),
                      Detection(0, 0.9, (float("inf"), 0, 2, 2)),
                      Detection(0, 0.9, (3, 0, 2, 2)),
                      Detection(0, 0.9, (20, 0, 21, 2))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(self.build(self.factory(detections))[0], 0)
            self.assertEqual(self.compare(factory=self.factory(detections))[0], 0)
        manifest, folder = self.manifest()
        self.assertEqual([r["box"] for r in manifest["regions"]], [[0, 1, 10, 8], [2, 3, 3, 4]])
        for record, size in zip(manifest["regions"], [(10, 7), (1, 1)], strict=True):
            with Image.open(folder / record["crop_path"]) as crop:
                self.assertEqual(crop.size, size)
        messages = "\n".join(str(w.message) for w in caught)
        for reason in ["non-finite confidence", "non-finite coordinates", "unordered coordinates", "empty box after clamping"]:
            self.assertIn(reason, messages)
        self.assertIn("reference:desired/kitchen/photo.png", messages)
        self.assertIn("candidate:rental-001/kitchen/photo.png", messages)

    def test_exif_orientation_rgb_and_hash_survive_evidence(self):
        for root, apartment in [(self.references, "desired"), (self.listings, "rental-001")]:
            path = root / apartment / "kitchen/photo.png"
            path.unlink()
            image = Image.new("L", (10, 8), 73)
            exif = Image.Exif()
            exif[274] = 6
            image.save(path.with_suffix(".jpg"), exif=exif)
        factory = self.factory([Detection(0, 0.5, (0, 0, 8, 10))])
        self.assertEqual(self.build(factory)[0], 0)
        manifest, folder = self.manifest()
        photo = manifest["regions"][0]["photo"]
        self.assertEqual((photo["width"], photo["height"]), (8, 10))
        path = self.references / "desired/kitchen/photo.jpg"
        digest = sha256(path.read_bytes()).hexdigest()
        self.assertEqual(photo["sha256"], digest)
        with Image.open(folder / manifest["regions"][0]["crop_path"]) as crop:
            self.assertEqual((crop.mode, crop.size, crop.getpixel((0, 0))), ("RGB", (8, 10), (73, 73, 73)))
        status, output, _ = self.compare(factory=factory)
        self.assertEqual(status, 0)
        self.assertIn("oriented size 8x10", output)
        self.assertIn(digest, output)
        self.assertEqual(sha256(path.read_bytes()).hexdigest(), digest)

    def test_candidate_catalog_is_fresh_and_unreadable_inputs_are_reported(self):
        self.assertEqual(self.build(self.factory())[0], 0)
        empty = self.listings / "broken/kitchen"
        empty.mkdir(parents=True)
        (empty / "bad.png").write_bytes(b"bad")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            status, output, _ = self.compare(factory=self.factory())
        self.assertEqual(status, 0)
        self.assertIn("Skipped: 1", output)
        self.assertIn("unreadable photo", str([str(w.message) for w in caught]))
        shutil.move(self.listings / "rental-001", self.listings / "renamed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = self.compare(factory=self.factory())[1]
        self.assertIn("Listing: renamed", output)
        self.assertNotIn("rental-001", output)
        (self.references / "desired/kitchen/bad.png").write_bytes(b"bad")
        self.assertIn("reference:desired/kitchen/bad.png: unreadable", self.build(self.factory())[2])

    def test_empty_detections_keep_inventory_and_unknown_class_is_fatal(self):
        blank = self.references / "desired/kitchen/blank.png"
        Image.new("RGB", (3, 3)).save(blank)
        factory = lambda: GeometryExtractor(
            lambda image: [] if image.width == 3 else [Detection(0, 0.5, (0, 0, 1, 1))],
            lambda crop: np.array([1, 0], dtype=np.float32),
            classes={0: "sink"}, dimension=2, configuration={})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(self.build(factory)[0], 0)
        self.assertIn("no valid target objects", str([str(w.message) for w in caught]))
        self.assertEqual(len(self.manifest()[0]["inventory"]), 2)
        self.assertEqual(len(self.manifest()[0]["regions"]), 1)
        self.assertIn("incompatible detector class", self.build(self.factory([Detection(99, 0.1, (0, 0, 1, 1))]))[2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status, _, errors = self.build(self.factory([]))
        self.assertEqual(status, 1)
        self.assertIn("no valid target objects", errors)


    def test_candidate_additions_edits_removals_and_empty_detections(self):
        factory = lambda: GeometryExtractor(
            lambda image: [] if image.getpixel((0, 0)) == (0, 0, 0) else [Detection(0, 0.5, (0, 0, 1, 1))],
            lambda crop: np.array([1, 0], dtype=np.float32),
            classes={0: "sink"}, dimension=2, configuration={})
        self.assertEqual(self.build(factory)[0], 0)
        folder = self.listings / "added/bathroom"
        folder.mkdir(parents=True)
        photo = folder / "new.png"
        Image.new("RGB", (2, 2), (100, 100, 100)).save(photo)
        self.assertIn("Evaluated: 2", self.compare(factory=factory)[1])
        Image.new("RGB", (2, 2)).save(photo)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            status, output, _ = self.compare(factory=factory)
        self.assertEqual(status, 0)
        self.assertIn("Evaluated: 2; Selected: 1; Below threshold: 1", output)
        self.assertIn("candidate:added/bathroom/new.png: no valid target objects", str([str(w.message) for w in caught]))
        photo.unlink()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIn("Evaluated: 1; Selected: 1; Below threshold: 0; Skipped: 1", self.compare(factory=factory)[1])

    def test_shared_configurable_cutoff_includes_equality(self):
        factory = self.factory([Detection(0, 0.7, (0, 0, 1, 1)), Detection(0, 0.69, (1, 1, 2, 2))], cutoff=0.7)
        self.assertEqual(self.build(factory)[0], 0)
        self.assertEqual(len(self.manifest()[0]["regions"]), 1)
        self.assertEqual(self.compare(factory=factory)[0], 0)
        self.assertIn("Incompatible reference library", self.compare(factory=self.factory(cutoff=0.6))[2])

"""Model-free contracts at the library and build/compare workflow seams."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from rank_homes.catalog import load_catalog
from rank_homes.references import ReferenceWriter, load_reference_library, save_reference_library
from rank_homes.workflows import build_references, compare_listings
from test_reference_search import FixtureExtractor


class ReferenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="rank-homes-integrity-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.library = self.root / "library"
        self.source = self.images / "desired/kitchen/photo.png"
        self.source.parent.mkdir(parents=True)
        Image.new("RGB", (10, 8), (40, 80, 120)).save(self.source)
        self.extractor = FixtureExtractor()

    def build(self):
        return build_references(self.images, self.library, self.extractor)

    def load(self):
        return load_reference_library(self.library, load_catalog(self.images, "reference"),
                                      self.extractor.configuration, self.extractor.dimension)

    def generation(self):
        pointer = json.loads((self.library / "CURRENT.json").read_text())
        return self.library / "generations" / pointer["generation"]

    def test_exact_executing_dependency_versions_are_compatibility_inputs(self):
        self.build()
        with patch("importlib.metadata.version", return_value="different-version"):
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.load()

    def test_missing_extraction_metadata_cannot_publish(self):
        self.extractor.configuration = {"dimension": 2}
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.build()
        self.assertFalse((self.library / "CURRENT.json").exists())

    def test_presentation_and_listing_threshold_do_not_affect_compatibility(self):
        self.build()
        self.extractor.configuration = {**self.extractor.configuration,
            "listing_threshold": 0.9, "presentation": "compact"}
        self.assertEqual(len(self.load().regions), 1)

    def test_writer_exclusion_starts_before_extraction(self):
        owner = self
        class ContendingExtractor(FixtureExtractor):
            def analyze(self, photo, image):
                with owner.assertRaisesRegex(ValueError, "writer|build.*progress"):
                    build_references(owner.images, owner.library, FixtureExtractor())
                return super().analyze(photo, image)
        build_references(self.images, self.library, ContendingExtractor())
        self.assertEqual(len(self.load().regions), 1)

    def test_interrupted_pointer_replacement_keeps_previous_generation(self):
        self.build()
        before = self.load()
        pointer = (self.library / "CURRENT.json").read_bytes()
        with patch.object(Path, "replace", side_effect=KeyboardInterrupt("interrupted publication")):
            with self.assertRaises(KeyboardInterrupt):
                self.build()
        self.assertEqual((self.library / "CURRENT.json").read_bytes(), pointer)
        np.testing.assert_array_equal(self.load().vectors, before.vectors)
        self.build()
        self.assertEqual(len(list((self.library / "generations").iterdir())), 3)

    def test_generation_is_reloaded_and_validated_before_publication(self):
        self.build()
        pointer = (self.library / "CURRENT.json").read_bytes()
        original_save = Image.Image.save
        def corrupt_save(image, path, *args, **kwargs):
            original_save(image, path, *args, **kwargs)
            Path(path).write_bytes(b"broken PNG output")
        with patch.object(Image.Image, "save", corrupt_save):
            with self.assertRaisesRegex(ValueError, "crop"):
                self.build()
        self.assertEqual((self.library / "CURRENT.json").read_bytes(), pointer)
        self.assertEqual(len(self.load().regions), 1)

    def test_root_relocation_preserves_compatibility(self):
        self.build()
        relocated = self.root / "relocated"
        shutil.move(str(self.images), relocated)
        self.images = relocated
        np.testing.assert_array_equal(self.load().vectors, [[1, 0]])

    def assert_preflight_rejection(self, message="reference|Reference|configuration"):
        class NoInference(FixtureExtractor):
            def analyze(self, photo, image):
                raise AssertionError("candidate inference must not run")
        extractor = NoInference()
        extractor.configuration = self.extractor.configuration
        with self.assertRaisesRegex(ValueError, message):
            compare_listings(self.library, self.images, self.root / "missing-listings", extractor)

    def test_all_inventory_changes_reject_before_candidate_inference(self):
        duplicate = self.source.with_name("duplicate.png")
        empty = self.source.with_name("empty.png")
        class SelectiveExtractor(FixtureExtractor):
            def analyze(self, photo, image):
                if photo.relative_path.endswith("empty.png"):
                    from rank_homes.domain import Analysis
                    return Analysis((), np.empty((0, 2), dtype=np.float32))
                return super().analyze(photo, image)
        self.extractor = SelectiveExtractor()
        shutil.copyfile(self.source, duplicate)
        Image.new("RGB", (10, 8), "white").save(empty)
        self.build()
        original = {p: p.read_bytes() for p in (self.source, duplicate, empty)}
        for path in original:
            for change in ("remove", "edit", "rename"):
                with self.subTest(path=path.name, change=change):
                    renamed = path.with_name("renamed.png")
                    if change == "remove":
                        path.unlink()
                    elif change == "edit":
                        Image.new("RGB", (10, 8), "red").save(path)
                    else:
                        path.rename(renamed)
                    self.assert_preflight_rejection("inventory changed")
                    renamed.unlink(missing_ok=True)
                    path.write_bytes(original[path])
        addition = self.source.with_name("addition.png")
        shutil.copyfile(self.source, addition)
        self.assert_preflight_rejection("inventory changed")

    def test_each_extraction_setting_requires_explicit_rebuild(self):
        self.build()
        changes = {"extractor_version": "v2", "detector_weights": "different", "detector_weights_sha256": "c" * 64,
            "classes": {0: "toilet"}, "detector_cutoff": 0.7, "geometry": "different-geometry",
            "embedding_weights": "different", "embedding_weights_sha256": "d" * 64,
            "embedding_revision": "v2", "processor": {"resize": [112, 112]}, "pooling": "mean",
            "dimension": 3, "normalization": "none", "dtype": "float16"}
        original = self.extractor.configuration
        for key, value in changes.items():
            with self.subTest(key=key):
                self.extractor.configuration = {**original, key: value}
                self.assert_preflight_rejection()
        self.extractor.configuration = original

    def test_matrix_corruption_and_invalid_vectors_reject_before_inference(self):
        self.build()
        folder = self.generation()
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        matrix = folder / "embeddings.npy"
        matrix.write_bytes(b"corrupt")
        self.assert_preflight_rejection("embeddings")
        for vectors in (np.array([[1, 0]], dtype=np.float64), np.array([[1, 0, 0]], dtype=np.float32),
                        np.array([[float("nan"), 0]], dtype=np.float32), np.zeros((1, 2), dtype=np.float32),
                        np.array([[1.00002, 0]], dtype=np.float32), np.empty((0, 2), dtype=np.float32)):
            with self.subTest(vectors=vectors):
                np.save(matrix, vectors, allow_pickle=False)
                manifest["matrix_sha256"] = sha256(matrix.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                self.assert_preflight_rejection("embedding")

    def test_unreadable_missing_and_unknown_libraries_have_actionable_errors(self):
        self.build()
        manifest_path = self.generation() / "manifest.json"
        original = manifest_path.read_text()
        for text in ('{"format_version": 999}', '[]', '{}', '{broken'):
            manifest_path.write_text(text)
            self.assert_preflight_rejection("Reference load failed.*rebuild")
        manifest_path.write_text(original)
        self.source.write_bytes(b"unreadable")
        self.assert_preflight_rejection("unreadable photo")
        self.images = self.root / "missing"
        self.assert_preflight_rejection("Source root not found")

    def test_reader_uses_captured_generation_during_pointer_switch(self):
        self.build()
        old_pointer = (self.library / "CURRENT.json").read_text()
        old_folder = self.generation()
        self.build()
        new_pointer = (self.library / "CURRENT.json").read_text()
        (self.library / "CURRENT.json").write_text(old_pointer)
        read_text = Path.read_text
        def switch_once(path, *args, **kwargs):
            text = read_text(path, *args, **kwargs)
            if path.name == "CURRENT.json":
                path.write_text(new_pointer)
                new_id = json.loads(new_pointer)["generation"]
                (self.library / "generations" / new_id / "embeddings.npy").write_bytes(b"broken")
            return text
        with patch.object(Path, "read_text", switch_once):
            np.testing.assert_array_equal(self.load().vectors, [[1, 0]])
        self.assertTrue(old_folder.is_dir())

    def test_process_exit_releases_writer_and_preserves_published_library(self):
        self.build()
        pointer = (self.library / "CURRENT.json").read_bytes()
        script = """
import os, sys
from pathlib import Path
from rank_homes.workflows import build_references
from test_reference_search import FixtureExtractor
class Interrupted(FixtureExtractor):
    def analyze(self, photo, image):
        os._exit(17)
build_references(Path(sys.argv[1]), Path(sys.argv[2]), Interrupted())
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.images), str(self.library)],
                                cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual((self.library / "CURRENT.json").read_bytes(), pointer)
        self.assertEqual(len(self.load().regions), 1)
        self.build()

    def test_another_process_cannot_build_while_writer_is_active(self):
        script = """
import sys
from pathlib import Path
from rank_homes.workflows import build_references
from test_reference_search import FixtureExtractor
try:
    build_references(Path(sys.argv[1]), Path(sys.argv[2]), FixtureExtractor())
except ValueError as error:
    print(error)
    sys.exit(23)
"""
        with ReferenceWriter(self.library):
            result = subprocess.run([sys.executable, "-c", script, str(self.images), str(self.library)],
                                    cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn("active writer", result.stdout)
        self.assertFalse((self.library / "CURRENT.json").exists())
        self.build()

    def test_unknown_or_missing_pointer_never_consumes_unpublished_generations(self):
        self.build()
        pointer_path = self.library / "CURRENT.json"
        valid = json.loads(pointer_path.read_text())
        for pointer in ({**valid, "format_version": 999}, {**valid, "generation": "../elsewhere"},
                        {**valid, "generation": None}, {}, []):
            with self.subTest(pointer=pointer):
                pointer_path.write_text(json.dumps(pointer))
                self.assert_preflight_rejection("Reference load failed.*rebuild")
        pointer_path.unlink()
        self.assert_preflight_rejection("Reference load failed.*rebuild")

    def test_configuration_key_order_is_canonical(self):
        self.build()
        configuration = dict(reversed(list(self.extractor.configuration.items())))
        configuration["classes"] = {"0": "sink"}
        configuration["processor"] = dict(reversed(list(configuration["processor"].items())))
        self.extractor.configuration = configuration
        self.assertEqual(len(self.load().regions), 1)

    def test_unreadable_and_region_free_builds_never_publish(self):
        from rank_homes.domain import Analysis
        class EmptyExtractor(FixtureExtractor):
            def analyze(self, photo, image):
                return Analysis((), np.empty((0, 2), dtype=np.float32))
        self.extractor = EmptyExtractor()
        with self.assertRaisesRegex(ValueError, "no valid target objects"):
            self.build()
        self.assertFalse((self.library / "CURRENT.json").exists())
        self.extractor = FixtureExtractor()
        self.build()
        pointer = (self.library / "CURRENT.json").read_bytes()
        self.extractor = EmptyExtractor()
        with self.assertRaisesRegex(ValueError, "no valid target objects"):
            self.build()
        self.source.write_bytes(b"unreadable")
        with self.assertRaisesRegex(ValueError, "unreadable photo"):
            self.build()
        self.assertEqual((self.library / "CURRENT.json").read_bytes(), pointer)

    def test_changed_sources_during_extraction_cannot_publish(self):
        self.build()
        pointer = (self.library / "CURRENT.json").read_bytes()
        source = self.source
        class EditingExtractor(FixtureExtractor):
            def analyze(self, photo, image):
                Image.new("RGB", (10, 8), "red").save(source)
                return super().analyze(photo, image)
        self.extractor = EditingExtractor()
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            self.build()
        self.assertEqual((self.library / "CURRENT.json").read_bytes(), pointer)

    def test_region_class_and_confidence_match_extraction_configuration(self):
        self.build()
        manifest_path = self.generation() / "manifest.json"
        original = manifest_path.read_text()
        for change in ({"object_class": "unknown"}, {"confidence": 0.2}):
            manifest = json.loads(original)
            manifest["regions"][0].update(change)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "class|confidence"):
                self.load()

    def test_manifest_evidence_is_validated_against_inventory_and_crop(self):
        mutations = {
            "dimension": lambda m: m.update(dimension=99),
            "configuration": lambda m: m["configuration"].update(dtype="float16"),
            "source": lambda m: m["regions"][0]["photo"].update(sha256="0" * 64),
            "bounds": lambda m: m["regions"][0].update(box=[1, 2, 99, 6]),
            "integer bounds": lambda m: m["regions"][0].update(box=[1.5, 2, 7, 6]),
            "confidence": lambda m: m["regions"][0].update(confidence=float("nan")),
            "class": lambda m: m["regions"][0].update(object_class=""),
            "crop path": lambda m: m["regions"][0].update(crop_path="../elsewhere.png"),
            "row": lambda m: m["regions"][0].update(row=1),
            "empty": lambda m: m.update(regions=[]),
        }
        self.build()
        manifest_path = self.generation() / "manifest.json"
        original = manifest_path.read_text()
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                manifest = json.loads(original)
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "Reference|reference|embedding|Incompatible"):
                    self.load()
        manifest_path.write_text(original)

    def test_crop_decode_and_dimensions_are_checked_even_with_matching_hash(self):
        self.build()
        folder = self.generation()
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        crop_path = folder / manifest["regions"][0]["crop_path"]
        for content in [b"not a PNG", None]:
            with self.subTest(content=content):
                if content is None:
                    Image.new("RGB", (1, 1)).save(crop_path)
                else:
                    crop_path.write_bytes(content)
                manifest["regions"][0]["crop_sha256"] = sha256(crop_path.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "crop"):
                    self.load()

    def test_invalid_crop_cannot_replace_published_library(self):
        self.build()
        before = self.load()
        pointer = (self.library / "CURRENT.json").read_bytes()
        with Image.new("RGB", (1, 1)) as wrong_crop:
            with self.assertRaisesRegex(ValueError, "crop.*size|crop.*dimensions"):
                save_reference_library(self.library, before, load_catalog(self.images, "reference"),
                                       self.extractor.configuration, [wrong_crop], 2)
        self.assertEqual((self.library / "CURRENT.json").read_bytes(), pointer)
        np.testing.assert_array_equal(self.load().vectors, before.vectors)


if __name__ == "__main__":
    unittest.main()

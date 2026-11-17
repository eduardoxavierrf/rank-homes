"""Complete ranking output through real catalog/library workflows and fixture inference."""
import math
import unittest
import warnings
import weakref

import numpy as np
from PIL import Image

import test_reference_search as search
from rank_homes.domain import Analysis, Region


class EvidenceExtractor:
    dimension = 2
    configuration = {**search.FixtureExtractor.configuration,
                     "extractor_version": "ranking-evidence-v1", "classes": {0: "sink", 1: "cabinet"}}

    def __init__(self):
        self.candidate_photos = []

    def analyze(self, photo, image):
        if photo.side == "candidate":
            self.candidate_photos.append(photo.relative_path)
        # Class winners come from distinct photos; path order breaks equal photo scores.
        label = "cabinet" if photo.relative_path.endswith("z.png") else "sink"
        score = 1. if photo.side == "reference" else .9
        region = Region(f"{photo.photo_id}:{label}", photo, label, .9, (1, 2, 7, 6))
        return Analysis((region,), np.array([[score, math.sqrt(1 - score * score)]], dtype=np.float32))


class RankingCommandTests(unittest.TestCase):
    setUp = search.ReferenceSearchTests.setUp
    command = search.ReferenceSearchTests.command
    build = search.ReferenceSearchTests.build
    compare = search.ReferenceSearchTests.compare

    def test_changed_candidate_bytes_are_fatal_even_when_no_longer_decodable(self):
        self.photo(self.listings, "A-first/kitchen/photo.png", (80, 20, 40))
        changed = self.listings / "rental-001/kitchen/photo.png"

        class Mutating(EvidenceExtractor):
            def analyze(self, photo, image):
                if photo.side == "candidate" and photo.apartment == "A-first":
                    changed.write_bytes(b"changed and corrupt")
                return super().analyze(photo, image)

        self.assertEqual(self.build(Mutating)[0], 0)
        status, output, errors = self.compare(factory=Mutating)
        self.assertEqual(status, 1)
        self.assertIn("Candidate source changed", errors)
        self.assertNotIn("Evaluated:", output)
        self.assertNotIn("Listing:", output)

    def test_warning_categories_are_independent_of_user_paths(self):
        broken = self.listings / "duplicate of rental/kitchen/broken.png"
        broken.parent.mkdir(parents=True)
        broken.write_bytes(b"broken")
        self.assertEqual(self.build(EvidenceExtractor)[0], 0)
        status, output, errors = self.compare(factory=EvidenceExtractor)
        self.assertEqual(status, 0, errors)
        self.assertIn("unreadable inputs=1", output)
        self.assertNotIn("ignored duplicate photos=", output)

    def test_candidate_arrays_and_pixels_are_released_between_listings(self):
        self.photo(self.listings, "second/kitchen/photo.png", (80, 40, 20))
        test = self

        class Lifetime(EvidenceExtractor):
            previous = None
            previous_image = None

            def analyze(self, photo, image):
                if photo.side == "candidate" and self.previous is not None:
                    test.assertIsNone(self.previous())
                    test.assertIsNone(self.previous_image())
                result = super().analyze(photo, image)
                if photo.side == "candidate":
                    self.previous = weakref.ref(result.vectors)
                    self.previous_image = weakref.ref(image)
                return result

        self.assertEqual(self.build(Lifetime)[0], 0)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(self.compare(factory=Lifetime)[0], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_reference_failures_preserve_generation_and_allow_whole_restart(self):
        self.assertEqual(self.build(EvidenceExtractor)[0], 0)
        before = {p: p.read_bytes() for p in self.library.rglob("*") if p.is_file()}
        for failure in (MemoryError("injected OOM"), RuntimeError("model failed"), KeyboardInterrupt()):
            class Failed(EvidenceExtractor):
                def analyze(self, photo, image):
                    raise failure
            status, output, errors = self.build(Failed)
            self.assertEqual(status, 1)
            self.assertIn("reference:desired/kitchen/photo.png", errors)
            self.assertNotIn("Published", output)
            self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
        self.assertEqual(self.build(EvidenceExtractor)[0], 0)

    def test_zero_evidence_and_all_skipped_summaries(self):
        self.assertEqual(self.build(EvidenceExtractor)[0], 0)

        class Empty(EvidenceExtractor):
            def analyze(self, photo, image):
                return Analysis((), np.empty((0, 2), dtype=np.float32))

        status, output, errors = self.compare(factory=Empty)
        self.assertEqual(status, 0, errors)
        self.assertIn("Evaluated: 1; Selected: 0; Below threshold: 1; Skipped: 0", output)
        self.assertIn("no detections=1", output)
        self.assertIn("missing comparable coverage=1", output)
        (self.listings / "rental-001/kitchen/photo.png").write_bytes(b"broken")
        status, output, errors = self.compare(factory=Empty)
        self.assertEqual(status, 0, errors)
        self.assertIn("Evaluated: 0; Selected: 0; Below threshold: 0; Skipped: 1", output)
        self.assertIn("unreadable inputs=1", output)

    def test_fatal_failures_and_interruptions_have_no_completed_ranking(self):
        self.assertEqual(self.build(EvidenceExtractor)[0], 0)
        for failure in (MemoryError("injected OOM"), RuntimeError("model failed"), KeyboardInterrupt()):
            class Failed(EvidenceExtractor):
                def analyze(self, photo, image):
                    raise failure
            with self.subTest(failure=type(failure).__name__):
                status, output, errors = self.compare(factory=Failed)
                self.assertIn(status, (1, 130))
                self.assertIn("candidate:rental-001/kitchen/photo.png", errors)
                self.assertNotIn("Evaluated:", output)
                self.assertNotIn("Listing:", output)
        self.assertEqual(self.compare(factory=EvidenceExtractor)[0], 0)

    def photo(self, root, relative, color):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 8), color).save(path)

    def test_all_selected_details_and_matching_photo_ties_are_printed_before_exit(self):
        extractor = EvidenceExtractor()
        for root, names in [(self.references, ["desired"]), (self.listings, ["rental-001", "A-rental"])]:
            for apartment in names:
                self.photo(root, f"{apartment}/kitchen/z.png", (90, 20, 10))
                self.photo(root, f"{apartment}/bathroom/a.png", (20, 90, 10))
                self.photo(root, f"{apartment}/kitchen/a.png", (20, 10, 90))
        # Missing cabinet stays in diagnostics but is excluded from the average.
        (self.listings / "A-rental/kitchen/z.png").unlink()
        self.assertEqual(self.build(lambda: extractor)[0], 0)
        status, output, errors = self.compare("0.5", lambda: extractor)
        self.assertEqual(status, 0, errors)
        self.assertLess(output.index("Listing: A-rental"), output.index("Listing: rental-001"))
        first, second = output.split("Listing: ")[1:]
        for details, apartment in [(first, "A-rental"), (second, "rental-001")]:
            for expected in [str(self.listings / apartment), "Threshold: 0.5000", "Room bathroom: 0.9000",
                             "Class sink: 0.9000", "Class cabinet:", "box (1, 2, 7, 6)",
                             str(self.references / "desired/bathroom/a.png"),
                             f"Matching photo: {self.listings / apartment / 'kitchen/a.png'} (one object match)",
                             f"Matching photo: {self.listings / apartment / 'bathroom/a.png'} (one object match)"]:
                self.assertIn(expected, details)
        self.assertIn("Class cabinet: 0.0000", first)
        self.assertIn("Warning: missing comparable evidence", first)
        self.assertIn("Room kitchen: 0.9000", first)
        self.assertIn("Evaluated: 2; Selected: 2; Below threshold: 0; Skipped: 0", output)
        self.assertNotIn("%", output)

    def test_threshold_changes_reuse_references_and_reprocess_candidates(self):
        extractor = EvidenceExtractor()
        self.assertEqual(self.build(lambda: extractor)[0], 0)
        before = {p: p.read_bytes() for p in self.library.rglob("*") if p.is_file()}
        for threshold, selected in [("0.70", 1), ("0.99", 0)]:
            status, output, errors = self.compare(threshold, lambda: extractor)
            self.assertEqual((status, errors), (0, ""))
            self.assertIn(f"Selected: {selected}", output)
            if not selected:
                self.assertNotIn("Listing:", output)
        self.assertEqual(extractor.candidate_photos, ["rental-001/kitchen/photo.png"] * 2)
        self.assertEqual(before, {p: p.read_bytes() for p in self.library.rglob("*") if p.is_file()})

    def test_initial_cutoff_exposes_selected_score_that_rounds_to_cutoff(self):
        class NearCutoff(EvidenceExtractor):
            def analyze(self, photo, image):
                result = super().analyze(photo, image)
                value = 1. if photo.side == "reference" else .70001
                return Analysis(result.regions, np.array([[value, math.sqrt(1-value*value)]], dtype=np.float32))
        self.assertEqual(self.build(NearCutoff)[0], 0)
        status, output, errors = self.compare(factory=NearCutoff)
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("Score: 0.7000; Threshold: 0.7000", output)
        self.assertIn("Selected: 1", output)
        comparison = output.split("Comparison value: ")[1].splitlines()[0]
        value, cutoff = map(float, comparison.split(" > "))
        self.assertGreater(value, cutoff)
        self.assertLess(value, .70005)

    def test_selected_listing_reports_unreadable_photo_as_partial_coverage(self):
        self.assertEqual(self.build(EvidenceExtractor)[0], 0)
        (self.listings / "rental-001/kitchen/broken.png").write_bytes(b"broken image")
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            status, output, errors = self.compare(factory=EvidenceExtractor)
        self.assertEqual(status, 0, errors)
        self.assertIn("Selected: 1", output)
        self.assertIn("broken.png", errors)
        self.assertIn("partial coverage", errors)

    def test_equal_listing_scores_are_printed_in_apartment_name_order(self):
        self.photo(self.listings, "A-rental/kitchen/photo.png", (30, 40, 80))
        self.assertEqual(self.build(EvidenceExtractor)[0], 0)
        status, output, errors = self.compare(factory=EvidenceExtractor)
        self.assertEqual((status, errors), (0, ""))
        self.assertLess(output.index("Listing: A-rental"), output.index("Listing: rental-001"))
        self.assertIn("Selected: 2", output)

    def test_invalid_embedding_stops_command_with_photo_and_region_provenance(self):
        class InvalidCandidate(EvidenceExtractor):
            def analyze(self, photo, image):
                result = super().analyze(photo, image)
                if photo.side == "candidate":
                    return Analysis(result.regions, np.array([[float("nan"), 0]], dtype=np.float32))
                return result
        self.assertEqual(self.build(InvalidCandidate)[0], 0)
        status, output, errors = self.compare(factory=InvalidCandidate)
        self.assertEqual(status, 1)
        self.assertIn("candidate:rental-001/kitchen/photo.png:sink", errors)
        self.assertIn("box (1, 2, 7, 6)", errors)
        self.assertNotIn("Evaluated:", output)
        self.assertNotIn("Listing:", output)

"""Model-free ranking contracts using known geometric cosines."""
import math
import random
import unittest

import numpy as np

from rank_homes.domain import Analysis, Photo, Region
from rank_homes.ranking import rank_listing


def analysis(items, *, side="candidate", dtype=np.float32):
    """Items are (room, class, cosine, path, box); reference axis is (1, 0)."""
    regions, vectors = [], []
    for index, (room, label, cosine, path, box) in enumerate(items):
        photo = Photo(f"{side}:{path}", side, path.split('/')[0], room, path, "a" * 64, 20, 20)
        regions.append(Region(f"{photo.photo_id}:{index}", photo, label, 0.9, box))
        vectors.append([cosine, math.sqrt(max(0, 1 - cosine * cosine))])
    return Analysis(tuple(regions), np.array(vectors, dtype=dtype).reshape(-1, 2))


def item(room, label, score=1., path=None, box=(0, 0, 10, 10)):
    return room, label, score, path or f"home/{room}/{label}.png", box


class RankingTests(unittest.TestCase):
    def test_invalid_vectors_fail_with_source_even_without_comparable_pairs(self):
        references = analysis([item("kitchen", "sink")], side="reference")
        candidate = analysis([item("bathroom", "toilet")])
        for vector in ([0., 0.], [np.nan, 0.], [np.inf, 0.], [1e-13, 0.]):
            with self.subTest(vector=vector):
                broken = Analysis(candidate.regions, np.array([vector], dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "home/bathroom/toilet.png"):
                    rank_listing("home", references, broken)


class RankingContractTests(unittest.TestCase):
    def test_approved_score_fixtures(self):
        keys = [("kitchen", "sink"), ("kitchen", "cabinet"), ("bathroom", "sink"), ("bathroom", "toilet")]
        references = analysis([item(*key) for key in keys], side="reference")
        fixtures = [([.9, .8, .8, .7], .8), ([.9, .9, .2, .2], .55),
                    ([.9, .9, None, None], .45), ([.9, None, .8, .8], .625),
                    ([.8, .8, .6, .6], .70), ([.7001] * 4, .7001), ([None] * 4, 0.)]
        for values, expected in fixtures:
            with self.subTest(expected=expected):
                candidate = analysis([item(*key, value) for key, value in zip(keys, values) if value is not None])
                result = rank_listing("home", references, candidate)
                self.assertAlmostEqual(result.score, expected, places=7)
                self.assertEqual(len(result.matches), 4)
                self.assertEqual(sum(m.candidate is None for m in result.matches), values.count(None))

    def test_negative_pairs_keep_evidence_and_missing_pairs_do_not(self):
        refs = analysis([item("kitchen", c) for c in ("sink", "cabinet")], side="reference")
        candidates = analysis([item("kitchen", "sink", v, f"home/kitchen/{i}.png") for i, v in enumerate([-.2, -.1])])
        result = rank_listing("home", refs, candidates)
        self.assertEqual(result.score, 0)
        cabinet, sink = result.matches
        self.assertIsNone(cabinet.candidate)
        self.assertIsNotNone(sink.candidate)
        self.assertEqual(sink.candidate.photo.relative_path, "home/kitchen/0.png")

    def test_only_reference_rooms_and_classes_count_and_rooms_have_equal_weight(self):
        for reference_items, candidate_items, expected in [
            ([item("kitchen", "sink"), item("kitchen", "cabinet")],
             [item("kitchen", "sink", .9), item("kitchen", "cabinet", .8), item("bathroom", "sink")], .85),
            ([item("kitchen", "sink")] + [item("bathroom", c) for c in ("sink", "toilet", "wall")],
             [item("kitchen", "sink", .9)] + [item("bathroom", c, .3) for c in ("sink", "toilet", "wall")]
             + [item("kitchen", "extra")], .6),
            ([item("kitchen", "sink")], [item("bathroom", "sink")], 0),
        ]:
            with self.subTest(expected=expected):
                result = rank_listing("home", analysis(reference_items, side="reference"), analysis(candidate_items))
                self.assertAlmostEqual(result.score, expected, places=7)

    def test_alternatives_can_win_across_reference_apartments_and_candidate_photos(self):
        refs = analysis([item("kitchen", "sink", path="ref-a/kitchen/s.png"),
                         item("kitchen", "cabinet", path="ref-b/kitchen/c.png"),
                         item("kitchen", "sink", -1, "ref-b/kitchen/s.png")], side="reference")
        items = [item("kitchen", "sink", .95, "home/kitchen/s.png"),
                 item("kitchen", "cabinet", .8, "home/kitchen/c.png"),
                 item("kitchen", "sink", .4, "home/kitchen/other.png")]
        result = rank_listing("home", refs, analysis(items))
        self.assertAlmostEqual(result.score, .875, places=7)
        self.assertEqual({m.reference.photo.apartment for m in result.matches}, {"ref-a", "ref-b"})
        self.assertEqual(len({m.candidate.photo.relative_path for m in result.matches}), 2)
        repeated = rank_listing("home", refs, analysis(items + items))
        self.assertEqual(repeated.score, result.score)

    def test_pair_and_matching_photo_ties_follow_paths_then_numeric_boxes(self):
        from rank_homes.ranking import matching_photo
        refs = [item("kitchen", c, path=p, box=b)
                for c in ("cabinet", "sink") for p in ("ref/kitchen/z.png", "ref/kitchen/A.png")
                for b in ((10, 0, 20, 10), (2, 0, 12, 10))]
        candidates = [item("kitchen", c, path=p, box=b)
                      for c, p in (("cabinet", "home/kitchen/z.png"), ("sink", "home/kitchen/A.png"))
                      for b in ((10, 0, 20, 10), (2, 0, 12, 10))]
        for seed in range(6):
            random.Random(seed).shuffle(refs)
            random.Random(seed + 1).shuffle(candidates)
            result = rank_listing("home", analysis(refs, side="reference"), analysis(candidates))
            for match in result.matches:
                self.assertEqual(match.reference.photo.relative_path, "ref/kitchen/A.png")
                self.assertEqual(match.reference.box, (2, 0, 12, 10))
                self.assertEqual(match.candidate.box, (2, 0, 12, 10))
            winner = matching_photo(result, "kitchen")
            self.assertEqual(winner.candidate.photo.relative_path, "home/kitchen/A.png")
            self.assertIsNone(matching_photo(result, "bathroom"))

    def test_strict_selection_at_exact_and_adjacent_float64_values(self):
        from rank_homes.domain import ListingResult
        from rank_homes.ranking import select_listings
        cutoff = .70
        below, above = math.nextafter(cutoff, -math.inf), math.nextafter(cutoff, math.inf)
        results = [ListingResult(name, score, (), ()) for name, score in
                   [("equal", cutoff), ("below", below), ("z", above), ("A", above)]]
        self.assertEqual([r.apartment for r in select_listings(results, cutoff)], ["A", "z"])

    def test_float32_storage_is_renormalized_in_float64(self):
        refs = analysis([item("kitchen", "sink", .8)], side="reference")
        self.assertAlmostEqual(rank_listing("home", refs, refs).score, 1., places=15)
        candidate = analysis([item("kitchen", "sink", .8)])
        refs = analysis([item("kitchen", "sink")], side="reference")
        score = rank_listing("home", refs, candidate).score
        self.assertAlmostEqual(score, .7999999928474427, places=15)
        self.assertNotEqual(score, float(np.float32(.8)))

    def test_cosine_excursions_only_clamp_within_numeric_tolerance(self):
        from unittest.mock import patch
        refs = analysis([item("kitchen", "sink")], side="reference")
        candidates = analysis([item("kitchen", "sink")])
        # Inject arithmetic excursions: ordinary unit inputs cannot reliably produce these.
        for cosine, expected in [(1 + 5e-13, 1.), (-1 - 5e-13, 0.)]:
            with self.subTest(cosine=cosine), patch("numpy.dot", return_value=cosine):
                result = rank_listing("home", refs, candidates)
                self.assertEqual(result.score, expected)
                self.assertIsNotNone(result.matches[0].candidate)
        for cosine in (1 + 2e-12, -1 - 2e-12, float("nan")):
            with self.subTest(cosine=cosine), patch("numpy.dot", return_value=cosine):
                with self.assertRaisesRegex(ValueError, "home/kitchen/sink.png.*box"):
                    rank_listing("home", refs, candidates)

    def test_invalid_reference_and_empty_library_fail_clearly(self):
        refs = analysis([item("kitchen", "sink")], side="reference")
        invalid = Analysis(refs.regions, np.array([[0., 0.]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "reference:home/kitchen/sink.png"):
            rank_listing("home", invalid, analysis([]))
        with self.assertRaisesRegex(ValueError, "Reference library has no valid regions"):
            rank_listing("home", analysis([]), analysis([]))

    def test_path_separators_are_normalized_before_tie_comparison(self):
        refs = analysis([item("kitchen", "sink")], side="reference")
        candidates = analysis([item("kitchen", "sink", path="home/kitchen/z.png"),
                               item("kitchen", "sink", path=r"home\kitchen\A.png")])
        winner = rank_listing("home", refs, candidates).matches[0]
        self.assertEqual(winner.candidate.photo.relative_path, r"home\kitchen\A.png")

    def test_matching_photo_uses_pair_strength_rather_than_class_or_room_order(self):
        from rank_homes.ranking import matching_photo
        refs = analysis([item("kitchen", c) for c in ("cabinet", "sink")], side="reference")
        candidates = analysis([item("kitchen", "cabinet", .2), item("kitchen", "sink", .9)])
        result = rank_listing("home", refs, candidates)
        self.assertEqual(matching_photo(result, "kitchen").object_class, "sink")
        missing = rank_listing("home", refs, analysis([]))
        self.assertIsNone(matching_photo(missing, "kitchen"))


if __name__ == "__main__":
    unittest.main()

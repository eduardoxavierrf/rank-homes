"""Pure same-room, same-class cosine ranking."""
import math
from collections.abc import Iterable
import numpy as np
from .domain import Analysis, ListingResult, Match


def _pair_key(match: Match):
    assert match.candidate is not None and match.reference is not None
    return (-match.score, match.candidate.photo.relative_path.replace("\\", "/"),
            match.reference.photo.relative_path.replace("\\", "/"),
            match.candidate.box, match.reference.box)


def matching_photo(result: ListingResult, room: str) -> Match | None:
    """Return the strongest winning pair in a room, with the same tie order."""
    pairs = (m for m in result.matches if m.room == room and m.candidate is not None)
    return min(pairs, key=_pair_key, default=None)


def select_listings(results: Iterable[ListingResult], threshold: float = 0.70) -> list[ListingResult]:
    """Apply the strict cutoff before presentation and order exact ties by name."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Listing threshold must be finite and between zero and one")
    return sorted((r for r in results if r.score > threshold), key=lambda r: (-r.score, r.apartment))


def _validated_vectors(analysis: Analysis):
    vectors = analysis.vectors.astype(np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(analysis.regions) or vectors.shape[1] == 0:
        raise ValueError("Invalid ranking embedding dimensions or region alignment")
    norms = np.linalg.norm(vectors, axis=1)
    for region, vector, norm in zip(analysis.regions, vectors, norms):
        if not np.isfinite(vector).all() or not math.isfinite(float(norm)) or norm <= 1e-12:
            raise ValueError(f"Invalid ranking vector: {region.region_id} ({region.photo.relative_path}, box {region.box})")
    return vectors, norms


def rank_listing(apartment: str, references: Analysis, candidate: Analysis) -> ListingResult:
    reference_vectors, reference_norms = _validated_vectors(references)
    candidate_vectors, candidate_norms = _validated_vectors(candidate)
    if reference_vectors.shape[1] != candidate_vectors.shape[1]:
        raise ValueError(f"{apartment}: incompatible ranking embedding dimensions")
    if not references.regions:
        raise ValueError("Reference library has no valid regions for ranking")
    matches = []
    expected = sorted({(r.photo.room, r.object_class) for r in references.regions})
    for room, object_class in expected:
        pairs = []
        for ci, c in enumerate(candidate.regions):
            if (c.photo.room, c.object_class) != (room, object_class):
                continue
            for ri, r in enumerate(references.regions):
                if (r.photo.room, r.object_class) != (room, object_class):
                    continue
                cv = candidate_vectors[ci]
                rv = reference_vectors[ri]
                cosine = float(np.dot(cv, rv) / (candidate_norms[ci] * reference_norms[ri]))
                if not math.isfinite(cosine) or abs(cosine) > 1 + 1e-12:
                    raise ValueError(f"Invalid cosine: {c.region_id} ({c.photo.relative_path}, box {c.box}), "
                                     f"{r.region_id} ({r.photo.relative_path}, box {r.box})")
                score = max(0.0, min(1.0, cosine))
                pairs.append(Match(room, object_class, score, c, r))
        if pairs:
            winner = min(pairs, key=_pair_key)
            matches.append(winner)
        else:
            matches.append(Match(room, object_class, 0.0, None, None))
    rooms = []
    for room in sorted({m.room for m in matches}):
        observed = [m.score for m in matches if m.room == room and m.candidate is not None]
        rooms.append((room, math.fsum(observed) / len(observed) if observed else 0.0))
    score = math.fsum(score for _, score in rooms) / len(rooms)
    return ListingResult(apartment, score, tuple(matches), tuple(rooms))

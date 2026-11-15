"""Pure same-room, same-class cosine ranking."""
import math
import numpy as np
from .domain import Analysis, ListingResult, Match


def rank_listing(apartment: str, references: Analysis, candidate: Analysis) -> ListingResult:
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
                cv = candidate.vectors[ci].astype(np.float64)
                rv = references.vectors[ri].astype(np.float64)
                cosine = float(np.dot(cv, rv) / (np.linalg.norm(cv) * np.linalg.norm(rv)))
                if not math.isfinite(cosine) or abs(cosine) > 1 + 1e-12:
                    raise ValueError(f"Invalid cosine: {c.region_id}, {r.region_id}")
                score = max(0.0, min(1.0, cosine))
                key = (-score, c.photo.relative_path, r.photo.relative_path, c.box, r.box)
                pairs.append((key, Match(room, object_class, score, c, r)))
        if pairs:
            _, winner = min(pairs, key=lambda pair: pair[0])
            matches.append(winner)
        else:
            matches.append(Match(room, object_class, 0.0, None, None))
    rooms = tuple((room, math.fsum(m.score for m in matches if m.room == room) /
                   sum(m.room == room for m in matches)) for room in sorted({m.room for m in matches}))
    score = math.fsum(score for _, score in rooms) / len(rooms)
    return ListingResult(apartment, score, tuple(matches), rooms)

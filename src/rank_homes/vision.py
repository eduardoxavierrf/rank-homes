"""Composition boundary for local inference; real model integration is pending."""
from typing import Protocol
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import warnings
from numpy.typing import NDArray
import numpy as np
from PIL import Image
from .domain import Analysis, Photo, Region


class VisualExtractor(Protocol):
    dimension: int
    configuration: dict[str, object]

    def analyze(self, photo: Photo, image: Image.Image) -> Analysis: ...


def create_extractor() -> VisualExtractor:
    raise ValueError("Real visual extraction is not integrated yet. A usable local extractor is required; no model weights were loaded.")


def validate_analysis(analysis: Analysis, dimension: int) -> None:
    vectors = analysis.vectors
    if vectors.dtype != np.float32 or vectors.shape != (len(analysis.regions), dimension):
        raise ValueError("Invalid embedding dtype, dimension or region alignment")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(vectors).all() or not np.all(np.abs(norms - 1) <= 1e-5):
        raise ValueError("Invalid embedding: expected finite unit vectors")
    if len({region.region_id for region in analysis.regions}) != len(analysis.regions):
        raise ValueError("Duplicate region identities")


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


class GeometryExtractor:
    """Apply the shared pixel contract between detector and embedding adapters."""

    def __init__(self, detector: Callable[[Image.Image], Iterable[Detection]],
                 embedder: Callable[[Image.Image], NDArray[np.float32]], *,
                 classes: Mapping[int, str], dimension: int,
                 configuration: dict[str, object], cutoff: float = 0.5):
        if not classes or any(not label for label in classes.values()) or len(set(classes.values())) != len(classes):
            raise ValueError("Incompatible detector class map")
        if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError("Detector cutoff must be finite and between zero and one")
        self.detector = detector
        self.embedder = embedder
        self.classes = dict(classes)
        self.dimension = dimension
        self.cutoff = cutoff
        self.configuration = {**configuration, "classes": dict(classes), "detector_cutoff": cutoff,
                              "geometry": "exif-rgb-clamp-floor-ceil-exclusive-v1"}

    def analyze(self, photo: Photo, image: Image.Image) -> Analysis:
        regions, vectors = [], []
        for index, detection in enumerate(self.detector(image)):
            if detection.class_id not in self.classes:
                raise ValueError(f"{photo.photo_id}: incompatible detector class {detection.class_id}")
            if not math.isfinite(detection.confidence):
                warnings.warn(f"{photo.photo_id}: detection {index}: non-finite confidence", stacklevel=2)
                continue
            if detection.confidence < self.cutoff:
                continue
            x1, y1, x2, y2 = detection.box
            box = (0, 0, 0, 0)
            reason = None
            if not all(math.isfinite(value) for value in detection.box):
                reason = "non-finite coordinates"
            elif x1 >= x2 or y1 >= y2:
                reason = "unordered coordinates"
            if reason is None:
                box = (math.floor(max(0, min(image.width, x1))),
                       math.floor(max(0, min(image.height, y1))),
                       math.ceil(max(0, min(image.width, x2))),
                       math.ceil(max(0, min(image.height, y2))))
                if box[0] >= box[2] or box[1] >= box[3]:
                    reason = "empty box after clamping"
            if reason is not None:
                warnings.warn(f"{photo.photo_id}: detection {index}: {reason}", stacklevel=2)
                continue
            with image.crop(box) as crop:
                vectors.append(self.embedder(crop))
            regions.append(Region(f"{photo.photo_id}:{index}", photo,
                                  self.classes[detection.class_id], detection.confidence, box))
        analysis = Analysis(tuple(regions), np.asarray(vectors, dtype=np.float32) if vectors
                            else np.empty((0, self.dimension), dtype=np.float32))
        validate_analysis(analysis, self.dimension)
        return analysis

"""Composition boundary for local inference; real model integration is pending."""
from typing import Protocol
import numpy as np
from PIL import Image
from .domain import Analysis, Photo


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

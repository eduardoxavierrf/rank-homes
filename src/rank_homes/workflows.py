"""Coordinate source reading, extraction, persistence and pure ranking."""
from pathlib import Path
import warnings
import numpy as np
from .catalog import load_catalog, read_image, representative_photos
from .domain import Analysis, Photo
from .ranking import rank_listing
from .references import load_reference_library, save_reference_library
from .vision import VisualExtractor, validate_analysis


def combine(analyses: list[Analysis], dimension: int) -> Analysis:
    return Analysis(tuple(r for a in analyses for r in a.regions),
                    np.concatenate([a.vectors for a in analyses]) if analyses else np.empty((0, dimension), dtype=np.float32))


def analyze_photo(photo: Photo, image, extractor: VisualExtractor) -> Analysis:
    analysis = extractor.analyze(photo, image)
    validate_analysis(analysis, extractor.dimension)
    if not analysis.regions:
        warnings.warn(f"{photo.photo_id}: no valid target objects", stacklevel=2)
    return analysis


def build_references(images: Path, output: Path, extractor: VisualExtractor) -> str:
    photos = load_catalog(images, "reference")
    analyses, crops = [], []
    for photo in representative_photos(photos):
        with read_image(images / photo.relative_path) as image:
            analysis = analyze_photo(photo, image, extractor)
            analyses.append(analysis)
            crops.extend(image.crop(region.box) for region in analysis.regions)
    return save_reference_library(output, combine(analyses, extractor.dimension), photos,
                                  extractor.configuration, crops, extractor.dimension)


def compare_listings(references: Path, reference_images: Path, listings: Path, extractor: VisualExtractor):
    library = load_reference_library(references, load_catalog(reference_images, "reference"),
                                     extractor.configuration, extractor.dimension)
    photos = representative_photos(load_catalog(listings, "candidate"))
    results = []
    for apartment in sorted(p.name for p in listings.iterdir() if p.is_dir()):
        analyses = []
        for photo in photos:
            if photo.apartment != apartment:
                continue
            try:
                image = read_image(listings / photo.relative_path)
            except (OSError, ValueError) as error:
                warnings.warn(f"{photo.photo_id}: unreadable photo: {error}", stacklevel=2)
                continue
            with image:
                analyses.append(analyze_photo(photo, image, extractor))
        if not analyses:
            warnings.warn(f"{apartment}: skipped candidate without usable photos", stacklevel=2)
            continue
        results.append(rank_listing(apartment, library, combine(analyses, extractor.dimension)))
    return sorted(results, key=lambda result: (-result.score, result.apartment))

"""Coordinate source reading, extraction, persistence and pure ranking."""
from pathlib import Path
from hashlib import sha256
from io import BytesIO
import warnings
import numpy as np
from .catalog import load_catalog, read_image, representative_photos
from .domain import Analysis, Photo
from .ranking import rank_listing
from .references import ReferenceWriter, load_reference_library
from .extraction_configuration import extraction_configuration
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
    with ReferenceWriter(output) as writer:
        configuration = extraction_configuration(extractor.configuration, extractor.dimension)
        photos = load_catalog(images, "reference")
        analyses, crops = [], []
        try:
            for photo in representative_photos(photos):
                source = images / photo.relative_path
                content = source.read_bytes()
                if sha256(content).hexdigest() != photo.sha256:
                    raise ValueError(f"Reference source changed before extraction: {source}; repeat the complete build")
                with read_image(BytesIO(content)) as image:
                    if image.size != (photo.width, photo.height):
                        raise ValueError(f"Reference dimensions changed before extraction: {source}")
                    analysis = analyze_photo(photo, image, extractor)
                    analyses.append(analysis)
                    crops.extend(image.crop(region.box) for region in analysis.regions)
            # The initial discovery already reported layout warnings.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="reference:.*: photo outside apartment/room layout")
                current_photos = load_catalog(images, "reference")
            if photos != current_photos:
                raise ValueError("Reference source inventory changed during build; repeat the complete build")
            return writer.publish(combine(analyses, extractor.dimension), photos,
                                  configuration, crops, extractor.dimension)
        finally:
            for crop in crops:
                crop.close()


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
                warnings.warn(f"{photo.photo_id}: unreadable photo: {error}; partial coverage (photo ignored)", stacklevel=2)
                continue
            with image:
                analyses.append(analyze_photo(photo, image, extractor))
        if not analyses:
            warnings.warn(f"{apartment}: skipped candidate without usable photos", stacklevel=2)
            continue
        results.append(rank_listing(apartment, library, combine(analyses, extractor.dimension)))
    return sorted(results, key=lambda result: (-result.score, result.apartment))

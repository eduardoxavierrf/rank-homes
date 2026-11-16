"""Coordinate source reading, extraction, persistence and pure ranking."""
from pathlib import Path
from hashlib import sha256
from io import BytesIO
from collections.abc import Callable
import warnings
import numpy as np
from .catalog import load_catalog, read_image, representative_photos
from .domain import Analysis, Photo
from .diagnostics import DiagnosticWarning
from .ranking import rank_listing
from .references import ReferenceWriter, load_reference_library
from .extraction_configuration import extraction_configuration
from .vision import Extractor, validate_analysis


def combine(analyses: list[Analysis], dimension: int) -> Analysis:
    return Analysis(tuple(r for a in analyses for r in a.regions),
                    np.concatenate([a.vectors for a in analyses]) if analyses else np.empty((0, dimension), dtype=np.float32))


def analyze_photo(photo: Photo, image, extractor: Extractor) -> Analysis:
    try:
        analysis = extractor.analyze(photo, image)
        validate_analysis(analysis, extractor.dimension)
    except (Exception, KeyboardInterrupt) as error:
        raise RuntimeError(f"Extraction failed for {photo.photo_id} ({photo.relative_path}): "
                           f"{type(error).__name__}: {error}; restart the complete command") from error
    if not analysis.regions:
        warnings.warn(DiagnosticWarning(f"{photo.photo_id}: no valid target objects", "no detections"), stacklevel=2)
    return analysis


def build_references(images: Path, output: Path, extractor: Extractor) -> str:
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


def score_candidate(apartment: str, photos: tuple[Photo, ...], listings: Path,
                    extractor: Extractor, library: Analysis):
    """Release all candidate pixels and vectors when this listing returns."""
    analyses = []
    for photo in photos:
        try:
            content = (listings / photo.relative_path).read_bytes()
        except (OSError, ValueError) as error:
            warnings.warn(DiagnosticWarning(f"{photo.photo_id}: unreadable photo: {error}; partial coverage (photo ignored)", "unreadable inputs"), stacklevel=2)
            continue
        if sha256(content).hexdigest() != photo.sha256:
            raise ValueError(f"Candidate source changed before extraction: {photo.photo_id}; restart the complete command")
        try:
            image = read_image(BytesIO(content))
        except (OSError, ValueError) as error:
            warnings.warn(DiagnosticWarning(f"{photo.photo_id}: unreadable photo: {error}; partial coverage (photo ignored)", "unreadable inputs"), stacklevel=2)
            continue
        with image:
            if image.size != (photo.width, photo.height):
                raise ValueError(f"Candidate source changed before extraction: {photo.photo_id}; restart the complete command")
            analyses.append(analyze_photo(photo, image, extractor))
    if not analyses:
        warnings.warn(DiagnosticWarning(f"{apartment}: skipped candidate without usable photos", "skipped listings"), stacklevel=2)
        return None
    result = rank_listing(apartment, library, combine(analyses, extractor.dimension))
    for match in result.matches:
        if match.candidate is None:
            warnings.warn(DiagnosticWarning(f"{apartment}: {match.room}/{match.object_class}: missing comparable evidence; "
                          "physical presence unknown", "missing comparable coverage"), stacklevel=2)
    return result


def compare_listings(references: Path, reference_images: Path, listings: Path, extractor: Extractor,
                     progress: Callable[[str], None] | None = None):
    library = load_reference_library(references, load_catalog(reference_images, "reference"),
                                     extractor.configuration, extractor.dimension)
    photos = representative_photos(load_catalog(listings, "candidate"))
    results = []
    for apartment in sorted(p.name for p in listings.iterdir() if p.is_dir()):
        if progress is not None:
            progress(f"Processing candidate {apartment}")
        result = score_candidate(apartment, tuple(p for p in photos if p.apartment == apartment),
                                 listings, extractor, library)
        if result is not None:
            results.append(result)
    return sorted(results, key=lambda result: (-result.score, result.apartment))

"""Versioned reference generations and explicit source compatibility."""
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import cast
from uuid import uuid4
import numpy as np
from PIL import Image
from .domain import Analysis, Photo, Region
from .extraction_configuration import extraction_configuration, fingerprint
from .vision import validate_analysis


class ReferenceWriter:
    """Hold an OS-managed destination lock for the entire build.

    The lock file stays in place. Closing the handle (including process exit)
    releases exclusion, so an interrupted build never needs manual lock removal.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = None

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        writer = (self.path / ".build.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if writer.seek(0, os.SEEK_END) == 0:
                    writer.write(b"0")
                    writer.flush()
                writer.seek(0)
                msvcrt.locking(writer.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            writer.close()
            raise ValueError(f"Reference build already in progress at {self.path}; wait for the active writer") from error
        self._lock = writer
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def publish(self, analysis: Analysis, photos: tuple[Photo, ...], configuration: dict[str, object],
                crops: list[Image.Image], dimension: int) -> str:
        if self._lock is None:
            raise ValueError("Reference publication requires an active writer")
        return _save_generation(self.path, analysis, photos, configuration, crops, dimension)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid reference JSON object: {path}")
    return value


def _load_generation(generation: Path, photos: tuple[Photo, ...],
                     configuration: dict[str, object], dimension: int) -> Analysis:
    manifest = _read_json(generation / "manifest.json")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise ValueError("Unsupported reference manifest format; rebuild references")
    if (type(manifest["dimension"]) is not int or manifest["dimension"] != dimension
            or dimension <= 0 or manifest["fingerprint"] != fingerprint(configuration)
            or manifest["fingerprint"] != fingerprint(manifest["configuration"])):
        raise ValueError("Incompatible reference library extraction configuration/dimension; rebuild references")
    if manifest["inventory"] != [asdict(p) for p in photos]:
        raise ValueError("Reference source inventory changed; rebuild references")
    inventory = {photo.photo_id: photo for photo in photos}
    if len(inventory) != len(photos):
        raise ValueError("Duplicate reference photo identities")
    for photo in photos:
        parts = photo.relative_path.split("/")
        if (len(parts) != 3 or parts != [photo.apartment, photo.room, parts[-1]]
                or any(part in {"", ".", ".."} or "\\" in part for part in parts)
                or photo.room not in {"kitchen", "bathroom"} or photo.side != "reference"
                or photo.photo_id != f"reference:{photo.relative_path}"
                or type(photo.width) is not int or type(photo.height) is not int
                or min(photo.width, photo.height) <= 0 or len(photo.sha256) != 64
                or any(c not in "0123456789abcdef" for c in photo.sha256)):
            raise ValueError(f"Invalid reference inventory provenance/dimensions: {photo.relative_path}")
    matrix_path = generation / "embeddings.npy"
    if sha256(matrix_path.read_bytes()).hexdigest() != manifest["matrix_sha256"]:
        raise ValueError(f"Corrupt reference embeddings: {matrix_path}; rebuild references")
    regions = []
    classes = cast(dict[str, str], configuration["classes"])
    cutoff = cast(float, configuration["detector_cutoff"])
    for row, record in enumerate(manifest["regions"]):
        if type(record["row"]) is not int or record["row"] != row:
            raise ValueError(f"Invalid reference row alignment: row {row}")
        photo = Photo(**record["photo"])
        region = Region(record["region_id"], photo, record["object_class"],
                        record["confidence"], tuple(record["box"]))
        source = f"{photo.relative_path}, region {region.region_id}"
        if inventory.get(photo.photo_id) != photo or photo.side != "reference":
            raise ValueError(f"Invalid reference source provenance: {source}")
        if (not isinstance(region.region_id, str) or not region.region_id
                or not isinstance(region.object_class, str) or not region.object_class
                or region.object_class not in classes.values()
                or type(region.confidence) not in (int, float)
                or not math.isfinite(region.confidence)
                or not cutoff <= region.confidence <= 1):
            raise ValueError(f"Invalid reference class/confidence/identity: {source}")
        box = region.box
        if (len(box) != 4 or any(type(value) is not int for value in box)
                or not (0 <= box[0] < box[2] <= photo.width and 0 <= box[1] < box[3] <= photo.height)):
            raise ValueError(f"Invalid reference box/dimensions: {source}")
        if record["crop_path"] != f"crops/{row}.png":
            raise ValueError(f"Invalid reference crop path: {source}")
        crop_path = generation / record["crop_path"]
        if sha256(crop_path.read_bytes()).hexdigest() != record["crop_sha256"]:
            raise ValueError(f"Corrupt reference crop: {crop_path}; rebuild references")
        try:
            with Image.open(crop_path) as crop:
                crop.load()
                if (crop.format != "PNG" or crop.mode != "RGB"
                        or crop.size != (box[2] - box[0], box[3] - box[1])):
                    raise ValueError("crop dimensions, mode or format mismatch")
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid reference crop: {crop_path}: {error}; rebuild references") from error
        regions.append(region)
    if not regions:
        raise ValueError("Reference library has no valid target objects; rebuild references")
    analysis = Analysis(tuple(regions), np.load(matrix_path, allow_pickle=False))
    validate_analysis(analysis, dimension)
    return analysis


def save_reference_library(path: Path, analysis: Analysis, photos: tuple[Photo, ...],
                           configuration: dict[str, object], crops: list[Image.Image], dimension: int) -> str:
    with ReferenceWriter(path) as writer:
        return writer.publish(analysis, photos, configuration, crops, dimension)


def _save_generation(path: Path, analysis: Analysis, photos: tuple[Photo, ...],
                     configuration: dict[str, object], crops: list[Image.Image], dimension: int) -> str:
    configuration = extraction_configuration(configuration, dimension)
    validate_analysis(analysis, dimension)
    if not analysis.regions:
        raise ValueError("Reference library has no valid target objects")
    generation_id = uuid4().hex
    generation = path / "generations" / generation_id
    (generation / "crops").mkdir(parents=True)
    np.save(generation / "embeddings.npy", analysis.vectors, allow_pickle=False)
    records = []
    for row, (region, crop) in enumerate(zip(analysis.regions, crops, strict=True)):
        if len(region.box) != 4 or crop.mode != "RGB" or crop.size != (region.box[2] - region.box[0], region.box[3] - region.box[1]):
            raise ValueError(f"Reference crop dimensions/mode mismatch: {region.region_id}")
        crop_path = f"crops/{row}.png"
        crop.save(generation / crop_path, format="PNG")
        records.append({**asdict(region), "row": row, "crop_path": crop_path,
                        "crop_sha256": sha256((generation / crop_path).read_bytes()).hexdigest()})
    manifest = {"format_version": 1, "dimension": dimension,
                "fingerprint": fingerprint(configuration), "configuration": configuration,
                "inventory": [asdict(p) for p in photos], "regions": records,
                "matrix_sha256": sha256((generation / "embeddings.npy").read_bytes()).hexdigest()}
    (generation / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    _load_generation(generation, photos, configuration, dimension)
    temporary = path / f"CURRENT.{generation_id}.tmp"
    temporary.write_text(json.dumps({"format_version": 1, "generation": generation_id}), encoding="utf-8")
    temporary.replace(path / "CURRENT.json")
    return generation_id


def load_reference_library(path: Path, photos: tuple[Photo, ...],
                           configuration: dict[str, object], dimension: int) -> Analysis:
    try:
        configuration = extraction_configuration(configuration, dimension)
        pointer = _read_json(path / "CURRENT.json")
        generation_id = pointer["generation"]
        if (type(pointer["format_version"]) is not int or pointer["format_version"] != 1
                or not isinstance(generation_id, str) or len(generation_id) != 32
                or any(c not in "0123456789abcdef" for c in generation_id)):
            raise ValueError("Unsupported reference pointer format")
        return _load_generation(path / "generations" / generation_id, photos, configuration, dimension)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, EOFError) as error:
        raise ValueError(f"Reference load failed at {path}: {error}; rebuild references explicitly") from error

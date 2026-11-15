"""Versioned reference generations and explicit source compatibility."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image
from .domain import Analysis, Photo, Region
from .vision import validate_analysis


def fingerprint(configuration: dict[str, object]) -> str:
    return sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_reference_library(path: Path, analysis: Analysis, photos: tuple[Photo, ...],
                           configuration: dict[str, object], crops: list[Image.Image], dimension: int) -> str:
    validate_analysis(analysis, dimension)
    if not analysis.regions:
        raise ValueError("Reference library has no valid target objects")
    path.mkdir(parents=True, exist_ok=True)
    lock = path / ".build.lock"
    writer = lock.open("x")
    try:
        with writer:
            generation_id = uuid4().hex
            generation = path / "generations" / generation_id
            (generation / "crops").mkdir(parents=True)
            np.save(generation / "embeddings.npy", analysis.vectors, allow_pickle=False)
            records = []
            for row, (region, crop) in enumerate(zip(analysis.regions, crops, strict=True)):
                crop_path = f"crops/{row}.png"
                crop.save(generation / crop_path, format="PNG")
                records.append({**asdict(region), "row": row, "crop_path": crop_path,
                                "crop_sha256": sha256((generation / crop_path).read_bytes()).hexdigest()})
            manifest = {"format_version": 1, "dimension": dimension,
                        "fingerprint": fingerprint(configuration), "configuration": configuration,
                        "inventory": [asdict(p) for p in photos], "regions": records,
                        "matrix_sha256": sha256((generation / "embeddings.npy").read_bytes()).hexdigest()}
            (generation / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary = path / f"CURRENT.{generation_id}.tmp"
            temporary.write_text(json.dumps({"format_version": 1, "generation": generation_id}), encoding="utf-8")
            temporary.replace(path / "CURRENT.json")
            return generation_id
    finally:
        writer.close()
        lock.unlink()


def load_reference_library(path: Path, photos: tuple[Photo, ...],
                           configuration: dict[str, object], dimension: int) -> Analysis:
    pointer = json.loads((path / "CURRENT.json").read_text(encoding="utf-8"))
    generation_id = pointer["generation"]
    if pointer["format_version"] != 1 or len(generation_id) != 32 or any(c not in "0123456789abcdef" for c in generation_id):
        raise ValueError("Unsupported reference pointer")
    generation = path / "generations" / generation_id
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1 or manifest["fingerprint"] != fingerprint(configuration):
        raise ValueError("Incompatible reference library; rebuild references")
    if manifest["inventory"] != [asdict(p) for p in photos]:
        raise ValueError("Reference source inventory changed; rebuild references")
    matrix_path = generation / "embeddings.npy"
    if sha256(matrix_path.read_bytes()).hexdigest() != manifest["matrix_sha256"]:
        raise ValueError("Corrupt reference embeddings")
    regions = []
    for row, record in enumerate(manifest["regions"]):
        if record["row"] != row:
            raise ValueError("Invalid reference row alignment")
        crop_path = generation / "crops" / f"{row}.png"
        if sha256(crop_path.read_bytes()).hexdigest() != record["crop_sha256"]:
            raise ValueError("Corrupt reference crop")
        regions.append(Region(record["region_id"], Photo(**record["photo"]), record["object_class"],
                              record["confidence"], tuple(record["box"])))
    analysis = Analysis(tuple(regions), np.load(matrix_path, allow_pickle=False))
    validate_analysis(analysis, dimension)
    return analysis

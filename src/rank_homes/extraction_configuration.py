"""Canonical extraction identity, independent of ranking and presentation."""
from hashlib import sha256
from importlib import metadata
import json
import math


def fingerprint(configuration: dict[str, object]) -> str:
    return sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def extraction_configuration(configuration: dict[str, object], dimension: int) -> dict[str, object]:
    """Snapshot effective adapter settings and installed versions without importing models.

    Adapters supply actual weight hashes and processor settings, never model-name
    defaults. A missing distribution is explicit for model-free adapters; real
    adapters necessarily require their executing packages to be installed.
    """
    required = {
        "extractor_version", "detector_weights", "detector_weights_sha256", "classes",
        "detector_cutoff", "geometry", "embedding_weights", "embedding_weights_sha256",
        "embedding_revision", "processor", "pooling", "dimension", "normalization", "dtype",
    }
    missing = required - configuration.keys()
    if missing:
        raise ValueError(f"Invalid extraction configuration: missing {', '.join(sorted(missing))}")
    # JSON round-trip also freezes mutable dictionaries supplied by adapters.
    effective = json.loads(json.dumps({key: value for key, value in configuration.items()
        if key not in {"listing_threshold", "presentation", "device", "hardware", "dependencies"}}, allow_nan=False))
    for key in required - {"classes", "detector_cutoff", "processor", "dimension"}:
        if not isinstance(effective[key], str) or not effective[key]:
            raise ValueError(f"Invalid extraction configuration: {key}")
    for key in ("detector_weights_sha256", "embedding_weights_sha256"):
        digest = effective[key]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid extraction configuration: {key}")
    classes = effective["classes"]
    cutoff = effective["detector_cutoff"]
    if (not isinstance(classes, dict) or not classes
            or any(not isinstance(label, str) or not label for label in classes.values())
            or len(set(classes.values())) != len(classes)
            or type(cutoff) not in (int, float) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1
            or type(dimension) is not int or dimension <= 0 or effective["dimension"] != dimension
            or effective["dtype"] != "float32" or effective["normalization"] != "L2"
            or not isinstance(effective["processor"], dict) or not effective["processor"]):
        raise ValueError("Invalid extraction configuration: classes/cutoff/embedding settings")
    versions = {}
    for name in ("rfdetr", "torch", "torchvision", "transformers", "Pillow", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    effective["dependencies"] = versions
    return effective

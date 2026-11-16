"""Local model composition and the shared detection/crop geometry contract."""
from typing import Protocol, Any, cast
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import warnings
from pathlib import Path
from hashlib import file_digest
import os
import pickle
from numpy.typing import NDArray
import numpy as np
from PIL import Image
from .domain import Analysis, Photo, Region
from .diagnostics import DiagnosticWarning


class Extractor(Protocol):
    dimension: int
    configuration: dict[str, object]

    def analyze(self, photo: Photo, image: Image.Image) -> Analysis: ...


def create_extractor(*, detector_path: Path | None = None, embedding_path: Path | None = None,
                     cutoff: float = 0.5, device: str | None = None) -> Extractor:
    detector_path = Path(detector_path or "artifacts/runs/rooms-expanded-small/checkpoint_best_total.pth").resolve()
    if not detector_path.is_file():
        raise ValueError(f"Local detector checkpoint is missing: {detector_path}; stage the 23-class Small checkpoint first")
    return VisualExtractor(detector_path, Path(embedding_path or "artifacts/weights/dinov3-vits16-lvd"),
                           cutoff=cutoff, device=device)


def validate_analysis(analysis: Analysis, dimension: int) -> None:
    vectors = analysis.vectors
    if vectors.dtype != np.float32 or vectors.shape != (len(analysis.regions), dimension):
        raise ValueError("Invalid embedding dtype, dimension or region alignment")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    for region, vector, norm in zip(analysis.regions, vectors, norms):
        if not np.isfinite(vector).all() or not math.isfinite(float(norm)) or abs(norm - 1) > 1e-5:
            raise ValueError(f"Invalid embedding: expected finite unit vectors: {region.region_id} "
                             f"({region.photo.relative_path}, box {region.box})")
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
                warnings.warn(DiagnosticWarning(f"{photo.photo_id}: detection {index}: discarded box: non-finite confidence", "discarded boxes"), stacklevel=2)
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
                warnings.warn(DiagnosticWarning(f"{photo.photo_id}: detection {index}: discarded box: {reason}", "discarded boxes"), stacklevel=2)
                continue
            with image.crop(box) as crop:
                try:
                    vectors.append(self.embedder(crop))
                except Exception as error:
                    raise RuntimeError(f"Embedding failed: {photo.photo_id}: detection {index}, "
                                       f"box {box}: {type(error).__name__}: {error}") from error
            regions.append(Region(f"{photo.photo_id}:{index}", photo,
                                  self.classes[detection.class_id], detection.confidence, box))
        analysis = Analysis(tuple(regions), np.asarray(vectors, dtype=np.float32) if vectors
                            else np.empty((0, self.dimension), dtype=np.float32))
        validate_analysis(analysis, self.dimension)
        return analysis


CLASSES = ("bathtub", "cabinet", "chair", "coffee maker", "countertop", "dishwasher",
           "faucet", "floor", "microwave", "mirror", "oven", "range hood", "refrigerator",
           "shelf", "shower", "shower door", "shower stall", "sink", "stove", "table",
           "toilet", "wall", "window")


class VisualExtractor(GeometryExtractor):
    """Own one local Small detector and DINOv3 embedder for a complete command."""

    def __init__(self, detector_path: Path, embedding_path: Path, *, cutoff: float = 0.5,
                 device: str | None = None):
        if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError("Detector cutoff must be finite and between zero and one")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from rfdetr.detr import RFDETR
        from .embeddings import DinoV3Embedder

        self.device = str(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
        # Validate metadata before constructing any model. Never infer labels from
        # an external training script or silently reinterpret a legacy checkpoint.
        try:
            checkpoint = torch.load(detector_path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
            raise ValueError(f"Invalid local detector checkpoint: {detector_path}: {error}") from error
        if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_config"), dict)
                or not isinstance(checkpoint.get("args"), dict)):
            raise ValueError("Invalid local detector checkpoint metadata; expected saved model configuration and class names")
        config = checkpoint.get("model_config", {})
        args = checkpoint.get("args", {})
        if (checkpoint.get("model_name") != "RFDETRSmall" or config.get("model_name") != "RFDETRSmall"
                or config.get("encoder") != "dinov2_windowed_small" or config.get("resolution") != 512
                or config.get("num_classes") != 23 or args.get("class_names") != list(CLASSES)
                or config.get("segmentation_head") or config.get("use_grouppose_keypoints")):
            raise ValueError("Expected RF-DETR Small, 512 pixels and the canonical full 23-class mapping; legacy Nano is incompatible")
        self.model = cast(Any, RFDETR.from_checkpoint(str(detector_path), device=self.device, amp=False))
        if type(self.model).__name__ != "RFDETRSmall" or list(self.model.class_names) != list(CLASSES):
            raise ValueError("Loaded detector architecture or class mapping differs from the 23-class Small recipe")
        detector_module = self.model.model.model
        detector_module.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        detector_module.to(device=self.device, dtype=torch.float32).eval()
        embedder = DinoV3Embedder(embedding_path, device=self.device)
        with detector_path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        detector_config = self.model.model_config.model_dump(mode="json")
        # Training-host paths and device do not affect the extraction recipe.
        for key in ("pretrain_weights", "device"):
            detector_config.pop(key, None)
        configuration = {**embedder.configuration, "extractor_version": "small-dinov3-v1",
            "detector_weights": "RFDETRSmall", "detector_weights_sha256": digest,
            "detector_config": detector_config, "detector_resize_antialias": False,
            "detector_prefilter": "canonical-classes>-1", "detector_confidence_comparison": ">=",
            "device": self.device, "batch_size": 1}
        super().__init__(self._detect, embedder, classes=dict(enumerate(CLASSES)),
                         dimension=embedder.dimension, configuration=configuration, cutoff=cutoff)

    def _detect(self, image: Image.Image) -> Iterable[Detection]:
        import torch
        from rfdetr import ClassThresholds
        with torch.inference_mode(), torch.autocast(torch.device(self.device).type, enabled=False):
            # RF-DETR uses strict >. Keep its complete top-k result, including
            # confidence zero, and apply the inclusive cutoff in GeometryExtractor.
            # Per-class thresholds exclude the checkpoint's no-object logit slot.
            detections = self.model.predict(image, threshold=ClassThresholds({}, default=-1.0),
                                            antialias=False, include_source_image=False)
        return [Detection(int(label), float(confidence), cast(tuple[float, float, float, float], tuple(float(v) for v in box)))
                for label, confidence, box in zip(detections.class_id, detections.confidence, detections.xyxy)]

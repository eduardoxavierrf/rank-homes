"""Pinned local Transformers DINOv3 crop embeddings, independent of detection."""
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
ARTIFACT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors", "LICENSE.md", "README.md")


def record_artifacts(folder: Path, revision: str) -> dict:
    """Record an already acquired official snapshot; never download from inference."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("DINOv3 revision must be a full immutable commit SHA")
    missing = [name for name in ARTIFACT_FILES if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete local DINOv3 artifacts: {', '.join(missing)}")
    manifest = {"origin": MODEL_ID, "revision": revision, "license": "DINOv3 License",
                "files": {name: sha256((folder / name).read_bytes()).hexdigest() for name in ARTIFACT_FILES}}
    (folder / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class DinoV3Embedder:
    """One RGB crop in, one normalized 384-dimensional float32 CLS vector out."""

    dimension = 384

    def __init__(self, folder: Path, *, device: str | None = None):
        folder = folder.resolve()
        required = (*ARTIFACT_FILES, "artifacts.json")
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise ValueError(f"Incomplete local DINOv3 artifacts: {', '.join(missing)}; stage an approved snapshot first")
        manifest = json.loads((folder / "artifacts.json").read_text(encoding="utf-8"))
        if (manifest.get("origin") != MODEL_ID or manifest.get("license") != "DINOv3 License"
                or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", ""))):
            raise ValueError("Invalid DINOv3 artifact provenance")
        for name in ARTIFACT_FILES:
            if sha256((folder / name).read_bytes()).hexdigest() != manifest.get("files", {}).get(name):
                raise ValueError(f"DINOv3 artifact hash mismatch: {name}")

        # Set before importing HF; local_files_only also protects already-imported HF.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import DINOv3ViTImageProcessor, DINOv3ViTModel

        self.processor = DINOv3ViTImageProcessor.from_pretrained(str(folder), local_files_only=True)
        self.processor.do_resize = True
        self.processor.size = {"height": 224, "width": 224}
        self.processor.resample = Image.Resampling.BILINEAR
        setattr(self.processor, "do_center_crop", False)
        self.processor.do_rescale = True
        self.processor.do_normalize = True
        self.model, loading = cast(Any, DINOv3ViTModel.from_pretrained(
            str(folder), local_files_only=True, use_safetensors=True,
            dtype=torch.float32, output_loading_info=True))
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Incomplete or incompatible DINOv3 weights")
        config = self.model.config
        if (config.hidden_size != 384 or config.patch_size != 16 or config.num_channels != 3
                or config.num_hidden_layers != 12 or config.num_attention_heads != 6
                or config.num_register_tokens != 4 or config.intermediate_size != 1536):
            raise ValueError("Expected DINOv3 ViT-S/16 LVD configuration with 384 dimensions")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.configuration = {
            "embedding_weights": MODEL_ID, "embedding_weights_sha256": manifest["files"]["model.safetensors"],
            "embedding_revision": manifest["revision"], "embedding_artifacts": manifest,
            "processor": self.processor.to_dict(), "model_config": config.to_dict(),
            "pooling": "CLS", "dimension": self.dimension, "normalization": "L2",
            "dtype": "float32", "batch_size": 1, "device": str(self.device),
            "dependencies": {name: version(name) for name in
                             ("torch", "torchvision", "transformers", "Pillow", "numpy", "safetensors")},
        }

    def __call__(self, crop: Image.Image) -> NDArray[np.float32]:
        import torch

        with crop.convert("RGB") as rgb, torch.inference_mode():
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            vector = self.model(**inputs).pooler_output[0].to(dtype=torch.float32)
            norm = torch.linalg.vector_norm(vector)
            if (tuple(vector.shape) != (self.dimension,) or not torch.isfinite(vector).all()
                    or not torch.isfinite(norm) or norm <= 1e-12):
                raise ValueError("Invalid DINOv3 CLS vector: expected finite 384 components and norm > 1e-12")
            return (vector / norm).cpu().numpy().copy()

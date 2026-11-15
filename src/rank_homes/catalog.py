"""Read source identities without importing inference libraries."""
from hashlib import sha256
from pathlib import Path
from PIL import Image, ImageOps
from .domain import Photo


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_catalog(root: Path, side: str) -> tuple[Photo, ...]:
    if not root.is_dir():
        raise ValueError(f"Source root not found: {root}")
    photos = []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
    for apartment in sorted(root.iterdir()):
        if not apartment.is_dir():
            continue
        for room in ("bathroom", "kitchen"):
            folder = apartment / room
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in extensions:
                    continue
                relative = path.relative_to(root).as_posix()
                with read_image(path) as image:
                    width, height = image.size
                photos.append(Photo(f"{side}:{relative}", side, apartment.name, room,
                                    relative, sha256(path.read_bytes()).hexdigest(), width, height))
    return tuple(photos)

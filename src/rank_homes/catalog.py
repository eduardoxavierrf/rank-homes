"""Read source identities without importing inference libraries."""
from hashlib import sha256
from pathlib import Path
import warnings
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
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        if len(parts) != 3 or parts[1] not in {"kitchen", "bathroom"}:
            warnings.warn(f"{side}:{relative}: photo outside apartment/room layout", stacklevel=2)
            continue
        try:
            digest = sha256(path.read_bytes()).hexdigest()
            with read_image(path) as image:
                width, height = image.size
        except (OSError, ValueError) as error:
            message = f"{side}:{relative}: unreadable photo: {error}"
            if side == "reference":
                raise ValueError(message) from error
            warnings.warn(message, stacklevel=2)
            continue
        photos.append(Photo(f"{side}:{relative}", side, parts[0], parts[1],
                            relative, digest, width, height))
    return tuple(photos)


def representative_photos(photos: tuple[Photo, ...]) -> tuple[Photo, ...]:
    groups: dict[tuple[str, str, str, str], Photo] = {}
    for photo in sorted(photos, key=lambda p: p.relative_path):
        key = (photo.side, photo.apartment, photo.room, photo.sha256)
        if key in groups:
            warnings.warn(f"{photo.photo_id}: duplicate of {groups[key].relative_path}", stacklevel=2)
        else:
            groups[key] = photo
    return tuple(groups.values())

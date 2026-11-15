"""Data exchanged by search components; no filesystem or model dependencies."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Photo:
    photo_id: str
    side: str
    apartment: str
    room: str
    relative_path: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class Region:
    region_id: str
    photo: Photo
    object_class: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Analysis:
    regions: tuple[Region, ...]
    vectors: NDArray[np.float32]


@dataclass(frozen=True)
class Match:
    room: str
    object_class: str
    score: float
    candidate: Region | None
    reference: Region | None


@dataclass(frozen=True)
class ListingResult:
    apartment: str
    score: float
    matches: tuple[Match, ...]
    room_scores: tuple[tuple[str, float], ...]

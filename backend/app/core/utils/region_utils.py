"""Utilities for slide region sampling (training / inference patches)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SlideRegion:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: Dict) -> "SlideRegion":
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data.get("width", data.get("w", data.get("patch_size", 512)))),
            height=int(data.get("height", data.get("h", data.get("patch_size", 512)))),
        )

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def contains_point(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


def iter_patch_coords(
    region: SlideRegion,
    patch_size: int,
    max_patches: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Generate top-left coordinates of patches tiling the region."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    coords: List[Tuple[int, int]] = []
    y_end = region.y + region.height
    x_end = region.x + region.width

    for py in range(region.y, y_end, patch_size):
        for px in range(region.x, x_end, patch_size):
            if px + patch_size > x_end or py + patch_size > y_end:
                continue
            coords.append((px, py))

    if not coords:
        coords.append((region.x, region.y))

    if max_patches is not None and len(coords) > max_patches:
        rng = random.Random(seed)
        coords = rng.sample(coords, max_patches)

    return coords


def single_patch_region(x: int, y: int, patch_size: int) -> SlideRegion:
    return SlideRegion(x=x, y=y, width=patch_size, height=patch_size)

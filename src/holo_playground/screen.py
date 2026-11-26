"""Screenshots and Holo's coordinate space, shared by the grounding sweep and the agent."""

import base64
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

COORD_SCALE = 1000  # Holo answers in [0, 1000] on both axes, whatever the image size


def to_pixels(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Model coordinates in [0, COORD_SCALE] -> pixels of a width x height image."""
    return x / COORD_SCALE * width, y / COORD_SCALE * height


@dataclass
class Screenshot:
    path: Path
    width: int
    height: int
    data_uri: str

    @classmethod
    def load(cls, path: Path) -> "Screenshot":
        with Image.open(path) as img:
            (width, height), fmt = img.size, img.format
        if fmt is None:
            raise ValueError(f"unknown image format: {path}")
        data = base64.b64encode(path.read_bytes()).decode()
        return cls(path, width, height, f"data:{Image.MIME[fmt]};base64,{data}")

    def to_pixels(self, x: float, y: float) -> tuple[float, float]:
        return to_pixels(x, y, self.width, self.height)

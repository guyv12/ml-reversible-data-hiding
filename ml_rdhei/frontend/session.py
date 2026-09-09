from dataclasses import dataclass
from numpy import ndarray
from pathlib import Path

from backend.predictor.results import Prediction

@dataclass
class HideSession:
    source_path: str
    source_image: ndarray
    prediction: Prediction | None = None
    marked_image: ndarray | None = None

    @property
    def output_path(self) -> str:
        path = Path(self.source_path)
        return str(path.parent / f"processed_{path.name}")

    @property
    def image_format(self) -> str:
        path = Path(self.source_path)
        return str(path.suffix)
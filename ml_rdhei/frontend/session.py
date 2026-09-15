from dataclasses import dataclass
from numpy import ndarray
from pathlib import Path

from backend.predictor.results import Prediction

@dataclass
class HideSession:
    source_path: Path
    source_image: ndarray
    prediction: Prediction | None = None
    marked_image: ndarray | None = None

    @property
    def output_path(self) -> Path:
        return self.source_path.with_name(f"processed_{self.source_path.name}")

    @property
    def image_format(self) -> str:
        return self.source_path.suffix.lower()
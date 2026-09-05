from PySide6.QtWidgets import QFrame, QLabel, QFormLayout
from PySide6.QtCore import Qt

from backend.predictor.results import QualityMetrics

from frontend.utils import load_stylesheet

class QualityMetricsPanel(QFrame):
    _ROWS = (
        ("PSNR", "psnr", "{:.2f} dB"),
        ("SSIM", "ssim", "{:.4f}"),
        ("Payload capacity", "payload_capacity", "{} b"),
        ("Embedding rate", "embedding_rate", "{:.4f} bpp"),
    )

    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        self._labels: dict[str, QLabel] = {}

        for title, attr, _ in self._ROWS:
            self._labels[attr] = self._create_label()
            layout.addRow(QLabel(f"{title} "), self._labels[attr])

        load_stylesheet(self, "metrics.css")

    def _create_label(self) -> QLabel:
        label = QLabel("-")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)

        return label

    def set_metrics(self, metrics: QualityMetrics) -> None:
        for _, attr, fmt in self._ROWS:
            self._labels[attr].setText(fmt.format(getattr(metrics, attr)))

    def clear(self) -> None:
        for label in self._labels.values():
            label.setText("-")
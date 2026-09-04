from PySide6.QtWidgets import QFrame, QLabel, QFormLayout
from PySide6.QtCore import Qt

from frontend.utils import load_stylesheet

class QualityMetrics(QFrame):
    psnr: float | None = None
    ssim: float | None = None
    payload_capacity: int | None = None
    embedding_rate: float | None = None
    
    def __init__(self):
        super().__init__()

        layout = QFormLayout(self)
        self.psnr_label = self._create_label()
        self.ssim_label = self._create_label()
        self.payload_capacity_label = self._create_label()
        self.embedding_rate_label = self._create_label()

        layout.addRow(QLabel("PSNR "), self.psnr_label)
        layout.addRow(QLabel("SSIM "), self.ssim_label)
        layout.addRow(QLabel("Payload capacity "), self.payload_capacity_label)
        layout.addRow(QLabel("Embedding rate "), self.embedding_rate_label)

        load_stylesheet(self, "metrics.css")

    def _create_label(self) -> QLabel:
        label = QLabel("-")
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        return label

    def update_metrics(
        self,
        psnr: float | None = None,
        ssim: float | None = None,
        payload_capacity: int | None = None,
        embedding_rate: float | None = None
    ):
        self.psnr_label.setText("-" if psnr is None else f"{psnr:.2f} dB")
        self.ssim_label.setText("-" if ssim is None else f"{ssim:.4f}")
        self.payload_capacity_label.setText("-" if payload_capacity is None else f"{payload_capacity} b")
        self.embedding_rate_label.setText("-" if embedding_rate is None else f"{embedding_rate:.4f} bpp")
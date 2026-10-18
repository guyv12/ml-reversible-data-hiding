from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from bitarray import bitarray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from torch import Tensor


@dataclass(frozen=True)
class QualityMetrics:
	psnr: float
	ssim: float
	payload_capacity: int
	embedding_rate: float

	@property
	def allow_embedding(self) -> bool:
		return self.payload_capacity > 0

def mean_squared_error(img0: np.ndarray, img1: np.ndarray) -> float:
	return np.mean((img0 - img1) ** 2, dtype=np.float64)

def moving_average(image: np.ndarray, size: int) -> np.ndarray:
	pad = size // 2
	padded = np.pad(image.astype(np.float64), pad, mode="symmetric")
	csum = np.concatenate(([0.0], np.cumsum(padded)))
	return ((csum[size:] - csum[:-size]) / size).astype(image.dtype)
	
def peak_signal_noise_ratio(original: np.ndarray, predicted: np.ndarray, data_range: int) -> float:
	error = mean_squared_error(original, predicted)

	return 10 * np.log10((data_range ** 2) / error)
	
def structural_similarity(original: np.ndarray, predicted: np.ndarray, data_range: int, win_size: int = 7) -> float:
	ux = moving_average(original, win_size)
	uy = moving_average(predicted, win_size)
	uxx = moving_average(original * original, win_size)
	uyy = moving_average(predicted * predicted, win_size)
	uxy = moving_average(original * predicted, win_size)

	cov_norm = win_size / (win_size - 1)

	vx = cov_norm * (uxx - ux * ux)
	vy = cov_norm * (uyy - uy * uy)
	vxy = cov_norm * (uxy - ux * uy)

	C1 = (0.01 * data_range) ** 2
	C2 = (0.03 * data_range) ** 2

	S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))

	pad = (win_size - 1) // 2

	return S[pad:-pad].mean(dtype=np.float64)

def compute_metrics(
	original: Tensor,
	error_map: Tensor,
	mask: Tensor,
	ad_bits: int,
	bpp: int = 8
) -> QualityMetrics:    

	y = original.flatten()[~mask.flatten()].numpy().astype(np.float32)
	y_pred = y - error_map.numpy().astype(np.float32)
	
	data_range = 2 ** bpp - 1
	pixels = original.numel()
	capacity = pixels * bpp - ad_bits
	return QualityMetrics(
		psnr=peak_signal_noise_ratio(y, y_pred, data_range=data_range),
		ssim=structural_similarity(y, y_pred, data_range=data_range),
		payload_capacity=capacity//8,
		embedding_rate=capacity / pixels,
	)

def compute_dicom_metrics(
	original: Tensor,
	lsb_error_map: Tensor,
	mask: Tensor,
	ad_bits: int,
	bpp: int = 16
) -> QualityMetrics:    

	flat_original = original.flatten().int()
	flat_mask = mask.flatten()

	y_high = flat_original >> 8
	pred_high = y_high.new_full(y_high.shape, 15)
	
	y_low = flat_original & 0x00FF
	pred_low = y_low.clone()
	pred_low[~flat_mask] = y_low[~flat_mask] - lsb_error_map.flatten().int()

	y = y_high * 2 ** 8 + y_low
	y_pred = pred_high * 2 ** 8 + pred_low

	y = y.numpy().astype(np.float32)
	y_pred = y_pred.numpy().astype(np.float32)

	data_range = 2 ** bpp - 1
	pixels = original.numel()
	capacity = pixels * bpp - ad_bits
	return QualityMetrics(
		psnr=peak_signal_noise_ratio(y, y_pred, data_range=data_range),
		ssim=structural_similarity(y, y_pred, data_range=data_range),
		payload_capacity=capacity//8,
		embedding_rate=capacity / pixels,
	)
	
@dataclass(frozen=True)
class Prediction:
	ad: bitarray
	metrics: QualityMetrics
	bpp: int
	shape: tuple[int, int]

	@property
	def pixels(self) -> int:
		return self.shape[0] * self.shape[1]
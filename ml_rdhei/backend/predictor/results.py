from dataclasses import dataclass
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from torch import full_like, Tensor
from bitarray import bitarray

from backend.compressor.encryption import NONCE_BITS

@dataclass(frozen=True)
class QualityMetrics:
	psnr: float
	ssim: float
	payload_capacity: int
	embedding_rate: float

	@property
	def allow_embedding(self) -> bool:
		return self.payload_capacity > 0

def compute_quality_metrics(
	original: Tensor,
	error_map: Tensor,
	mask: Tensor,
	ad_bits: int,
	bpp: int = 8
) -> QualityMetrics:    

	y = original.squeeze().numpy().astype(np.float32)
	m = mask.numpy()

	y_pred = y.copy()
	y_pred[~m] = y[~m] - error_map.numpy().astype(np.float32)
	
	data_range = 2 ** bpp - 1
	pixels = original.numel()
	capacity = pixels * bpp - ad_bits - 2 * NONCE_BITS
	return QualityMetrics(
		psnr=peak_signal_noise_ratio(y, y_pred, data_range=data_range),
		ssim=structural_similarity(y, y_pred, data_range=data_range, 
		                           gaussian_weights=True, sigma=1.5, use_sample_covariance=False),
		payload_capacity=capacity//8,
		embedding_rate=capacity / pixels,
	)

def compute_dicom_quality_metrics(
	original: Tensor,
	lsb_error_map: Tensor,
	mask: Tensor,
	ad_bits: int, # plaintext AD length (before encryption, without nonce)
	bpp: int = 16
) -> QualityMetrics:    

	orig = original.squeeze().int()

	y_high = orig >> 8
	pred_high = y_high.clone()
	
	y_low = orig & 0x00FF
	pred_low = y_low.clone()
	pred_low[~mask] = y_low[~mask] - lsb_error_map.flatten().int()

	y = y_high * 2 ** 8 + y_low
	y_pred = pred_high * 2 ** 8 + pred_low

	y = y.numpy().astype(np.float32)
	y_pred = y_pred.numpy().astype(np.float32)

	data_range = 2 ** bpp - 1
	pixels = original.numel()
	capacity = pixels * bpp - ad_bits - 2 * NONCE_BITS
	return QualityMetrics(
		psnr=peak_signal_noise_ratio(y, y_pred, data_range=data_range),
		ssim=structural_similarity(y, y_pred, data_range=data_range, 
									gaussian_weights=True, sigma=1.5, use_sample_covariance=False),
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
from dataclasses import dataclass
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

@dataclass(frozen=True)
class QualityMetrics:
	psnr: float
	ssim: float
	payload_capacity: int
	embedding_rate: float

def compute_metrics(
	original,
	error_map,
	mask,
	ad_bits,
	bpp=8
) -> QualityMetrics:    

	y = original.flatten()[~mask.flatten()].numpy().astype(np.float32)
	y_pred = y - error_map.numpy().astype(np.float32)
	data_range = 2 ** bpp - 1
	pixels = original.numel()
	capacity = pixels * bpp - ad_bits
	return QualityMetrics(
		psnr=peak_signal_noise_ratio(y, y_pred, data_range=data_range),
		ssim=structural_similarity(y, y_pred, data_range=data_range),
		payload_capacity=capacity,
		embedding_rate=capacity / pixels,
	)
	
@dataclass(frozen=True)
class Prediction:
	ad: bitarray
	metrics: QualityMetrics
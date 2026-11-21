from dataclasses import dataclass
import numpy as np
from scipy.stats import chi2

@dataclass(frozen=True)
class EncryptionMetrics:
	npcr: float
	uaci: float
	information_entropy: float
	histogram_uniformity: float

def compute_encryption_metrics(original: np.ndarray, encrypted: np.ndarray, bpp: int) -> EncryptionMetrics:
	levels = 2 ** bpp
	diff = np.abs(original.astype(np.float64) - encrypted.astype(np.float64))
	hist = np.bincount(encrypted.ravel(), minlength=levels)
	p = hist[hist > 0] / encrypted.size
	expected = encrypted.size / levels
	chi2_stat = np.sum((hist - expected) ** 2) / expected

	return EncryptionMetrics(
		npcr=np.mean(original != encrypted) * 100,
		uaci=np.mean(diff) / (levels - 1) * 100,
		information_entropy=-np.sum(p * np.log2(p)),
		histogram_uniformity=chi2.sf(chi2_stat, levels - 1),
	)
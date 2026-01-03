from . import loader
from . import features
import torch
import time
import re


def test_dataloader_time() -> None:
    """
    time check for dataloader
    """
    
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    start_time = time.perf_counter()

    BOSSbase_train_loader, data_len = loader.get_loader("datasets/BOSSbase_512", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    # BOWS2_train_loader, data_len = data.get_loader("datasets/BOWS2_512/train", re.compile(r"[0-4][0-9]?[0-9]?[0-9]?\.pgm"))
    
    img_shape = next(iter(BOSSbase_train_loader)).shape[1:]
    all_imgs = torch.empty((data_len, *img_shape), dtype=torch.float32)

    for idx, batch in enumerate(BOSSbase_train_loader):
        start = idx * batch.shape[0]
        end = start + batch.shape[0]

        all_imgs[start:end] = batch

    end_time = time.perf_counter()

    assert (end_time - start_time) < 2., "Execution time longer than 2s"


def test_feature_extraction() -> None:
    """
    Simple unit test for the extract_features function, checking shapes and content
    for a controlled edge case.
    """

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    B, H, W, K = 2, 4, 4, 3
    
    batch = torch.ones((B, H, W), device=dev, dtype=torch.float)
    
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[1:3, 1:3] = True
    # Mask looks like:
    # [[F, F, F, F],
    #  [F, T, T, F],
    #  [F, T, T, F],
    #  [F, F, F, F]]
    
    # Number of masked pixels (P_known) per image: 4
    # Number of unmasked pixels (P_unknown) per image: 16 - 4 = 12
    N_unmasked = 12

    X, y = features.extract_features(batch, mask, K)


    assert X.shape == (B * N_unmasked, K * K), "X shape is incorrect"
    assert y.shape == (B, 12), "y shape is incorrect"

    expected_y = torch.ones((B, 12), device=dev)
    assert torch.allclose(y, expected_y), "y content should be all ones"
  
    # Simpler check: Every feature vector in X must have a sum of at least 0 and at most 4.
    max_sum = K * K 
    assert torch.all(X.sum(dim=1) <= 4.0), "Max patch sum should not exceed 4 (size of masked area)"
    assert torch.all(X.sum(dim=1) >= 0.0), "Patch sums must be non-negative"

    # Sum of the first feature vector should be 4.0
    assert X[0].sum() - 4.0 < 1e-8, "The patch for (0,0) should sum to 4"


# to run: pytest ml_rdhei/data/test.py -v
import torch
from functools import partial

from backend.predictor.pipelines import *
from backend.predictor.predictors import *
from backend.predictor.features import *
from backend.predictor.operations import *


def ad_unfold_ridge_border(batch: torch.Tensor, K: int = 5):
    _, H, W = batch.shape
    mask = reference_mask(H, W)

    feature_fn = partial(
        unfold_features, 
        K=K
    )
    predictor_fn = partial(
        ridge_prediction, 
        pred_fn=partial(dot_product_with_border, mask=mask)
    )

    yield from get_ad(batch, mask, feature_fn, predictor_fn)


def dicom_ad_unfold_ridge_border(batch: torch.Tensor, K: int = 5):
    _, H, W = batch.shape
    mask = reference_mask(H, W)

    feature_fn = partial(
        unfold_features, 
        K=K
    )
    predictor_fn = partial(
        ridge_prediction, 
        pred_fn=partial(dot_product_with_border, mask=mask)
    )

    yield from get_dicom_ad(batch, mask, feature_fn, predictor_fn)


def ad_mobilenetv2_ridge_border(batch: torch.Tensor):
    _, H, W = batch.shape
    mask = reference_mask(H, W)
    
    feature_fn = partial(
        cnn_features
    )
    predictor_fn = partial(
        ridge_prediction, 
        pred_fn=partial(dot_product_with_border, mask=mask)
    )
    
    yield from get_ad(batch, mask, feature_fn, predictor_fn)


def ad_cnn_ridge_border(batch: torch.Tensor):
    _, H, W = batch.shape
    mask = reference_mask(H, W)
        
    feature_fn = partial(
        cnn_features
    )
    predictor_fn = partial(
        ridge_prediction, 
        pred_fn=partial(dot_product_with_border, mask=mask)
    )
        
    yield from get_ad(batch, mask, feature_fn, predictor_fn)


# def ad_cnn(batch: torch.Tensor):
#     _, H, W = batch.shape
#     mask = reference_mask(H, W)
        
#     feature_fn = partial(
#         cnn_features
#     )
#     predictor_fn = partial(
#         ridge_prediction, 
#         pred_fn=partial(dot_product_with_border, mask=mask)
#     )
        
#     yield from get_ad(batch, mask, feature_fn, predictor_fn)

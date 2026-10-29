import re

from functools import partial

import backend.data.loader as dloader
import backend.predictor.predict as prediction
import backend.compressor.compress as compression
import backend.receiver.receive as receiver

from backend.predictor.results import compute_quality_metrics, compute_dicom_quality_metrics
from backend.models.models import *
from experiments.runners import *


# ----- Preprocessing tests -----

# --- Classic Ridge ---
# - pgm -
def test_preprocessing_ad_unfold_ridge_border(source_regex: re.Pattern):
    loader, _ = dloader.get_loader("../datasets/pgm/test", regex=source_regex)
    K = 5

    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.ad_unfold_ridge_border,
            K=K
        ),
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_quality_metrics,
        bpp=8,
    )

# - dcm -
def test_preprocessing_dicom_ad_unfold_ridge_border(source_regex: re.Pattern):
    loader, _ = dloader.get_dicom_loader("../datasets/dcm/test", regex=source_regex)
    K = 5

    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.dicom_ad_unfold_ridge_border,
            K=K
        ),
        prep_fn=prepare_dicom,
        compressor_fn=compression.compress_dicom_ad,
        metrics_fn=compute_dicom_quality_metrics,
        bpp=16,
    )

# --- CNN features + Ridge ---
# - pgm -
def test_preprocessing_ad_unet_mobilenet_v2_ridge(source_regex: re.Pattern):
    loader, _ = dloader.get_loader("../datasets/pgm/test", regex=source_regex, batch_size=1)
    K = 5
    model = get_mobilenet_v2_unet_model(classes=K ** 2)
    model.eval()
    save_model(model, "unet_mobilenetv2.pth")

    model_fn = partial(
        get_mobilenet_v2_unet_model,
        path="unet_mobilenetv2.pth",
        classes=K ** 2,
    )
    run_preprocessing(
        loader,
        predictor_fn=partial(prediction.ad_cnn_ridge, model_fn=model_fn),
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_quality_metrics,
        bpp=8,
    )

def test_preprocessing_ad_unet_resnet50_ridge(source_regex: re.Pattern):
    loader, _ = dloader.get_loader("../datasets/pgm/test", regex=source_regex, batch_size=1)
    K = 5
    model = get_resnet_50_unet_model(classes=K ** 2)
    model.eval()
    save_model(model, "unet_resnet50.pth")

    model_fn = partial(
        get_resnet_50_unet_model,
        path="unet_resnet50.pth",
        classes=K ** 2,
    )
    run_preprocessing(
        loader,
        predictor_fn=partial(prediction.ad_cnn_ridge, model_fn=model_fn),
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_quality_metrics,
        bpp=8,
    )

# - dcm -
def test_preprocessing_dicom_ad_unet_mobilenet_v2_ridge(source_regex: re.Pattern):
    loader, _ = dloader.get_dicom_loader("../datasets/dcm/test", regex=source_regex) # Keep batch size as 1 to avoid discrepancy
    K = 5
    model = get_mobilenet_v2_unet_model(classes=K ** 2)
    model.eval()
    save_model(model, "unet_mobilenetv2.pth") # save so it's the same for recovery

    model_fn = partial(
        get_mobilenet_v2_unet_model,
        path="unet_mobilenetv2.pth",
        classes=K ** 2
    )

    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.dicom_ad_cnn_ridge,
            model_fn=model_fn
        ),
        prep_fn=prepare_dicom,
        compressor_fn=compression.compress_dicom_ad,
        metrics_fn=compute_dicom_quality_metrics,
        bpp=16,
    )

def test_preprocessing_dicom_ad_unet_resnet50_ridge(source_regex: re.Pattern):
    loader, _ = dloader.get_dicom_loader("../datasets/dcm/test", regex=source_regex, batch_size=1) # Keep batch size as 1 to avoid discrepancy
    K = 5
    model = get_resnet_50_unet_model(classes=K ** 2)
    model.eval()
    save_model(model, "unet_resnet50.pth") # save so it's the same for recovery

    model_fn = partial(
        get_resnet_50_unet_model,
        path="unet_resnet50.pth",
        classes=K ** 2
    )

    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.dicom_ad_cnn_ridge,
            model_fn=model_fn
        ),
        prep_fn=prepare_dicom,
        compressor_fn=compression.compress_dicom_ad,
        metrics_fn=compute_dicom_quality_metrics,
        bpp=16,
    )

# --- Full CNN ---
# - pgm -
def test_preprocessing_ad_cnn(source_regex: re.Pattern):
    loader, _ = dloader.get_loader("../datasets/pgm/test", regex=source_regex)

    model_fn=partial(
        get_custom_model,
        path="cnn.pth",
    )
    
    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.ad_cnn,
            model_fn=model_fn
        ),
        prep_fn=prepare_cnn,
        compressor_fn=compression.compress_cnn_ad,
        metrics_fn=compute_quality_metrics,
        bpp=8,
    )

def test_preprocessing_ad_mobilenet_v2(source_regex: re.Pattern):
    loader, _ = dloader.get_loader("../datasets/pgm/test", regex=source_regex)
    model = get_mobilenet_v2_unet_model(classes=1)
    model.eval()
    save_model(model, "unet_mobilenetv2.pth")

    model_fn = partial(
        get_mobilenet_v2_unet_model,
        path="unet_mobilenetv2.pth",
        classes=1,
    )
    run_preprocessing(
        loader,
        predictor_fn=partial(prediction.ad_cnn, model_fn=model_fn),
        prep_fn=prepare_cnn,
        compressor_fn=compression.compress_cnn_ad,
        metrics_fn=compute_quality_metrics,
        bpp=8,
    )

def test_preprocessing_ad_resnet_50(source_regex: re.Pattern):
    loader, _ = dloader.get_loader("../datasets/pgm/test", regex=source_regex)
    model = get_resnet_50_unet_model(classes=1)
    model.eval()
    save_model(model, "unet_resnet50.pth")

    model_fn = partial(
        get_resnet_50_unet_model,
        path="unet_resnet50.pth",
        classes=1,
    )
    run_preprocessing(
        loader,
        predictor_fn=partial(prediction.ad_cnn, model_fn=model_fn),
        prep_fn=prepare_cnn,
        compressor_fn=compression.compress_cnn_ad,
        metrics_fn=compute_quality_metrics,
        bpp=8,
    )

# - dcm -
def test_preprocessing_dicom_ad_cnn(source_regex: re.Pattern):
    loader, _ = dloader.get_dicom_loader("../datasets/dcm/test", regex=source_regex)

    model_fn=partial(
        get_custom_model,
        path="cnn.pth",
    )
    
    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.dicom_ad_cnn,
            model_fn=model_fn
        ),
        prep_fn=prepare_dicom_cnn,
        compressor_fn=compression.compress_cnn_dicom_ad,
        metrics_fn=compute_dicom_quality_metrics,
        bpp=16,
    )

def test_preprocessing_dicom_ad_mobilenet_v2(source_regex: re.Pattern):
    loader, _ = dloader.get_dicom_loader("../datasets/dcm/test", regex=source_regex)
    model = get_mobilenet_v2_unet_model(classes=1)
    model.eval()
    save_model(model, "unet_mobilenetv2.pth") # save so it's the same for recovery

    model_fn=partial(
        get_mobilenet_v2_unet_model, 
        path="unet_mobilenetv2.pth",
        classes=1
    )
    
    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.dicom_ad_cnn,
            model_fn=model_fn
        ),
        prep_fn=prepare_dicom_cnn,
        compressor_fn=compression.compress_cnn_dicom_ad,
        metrics_fn=compute_dicom_quality_metrics,
        bpp=16,
    )

def test_preprocessing_dicom_ad_resnet_50(source_regex: re.Pattern):
    loader, _ = dloader.get_dicom_loader("../datasets/dcm/test", regex=source_regex)
    model = get_resnet_50_unet_model(classes=1)
    model.eval()
    save_model(model, "unet_resnet50.pth") # save so it's the same for recovery

    model_fn=partial(
        get_resnet_50_unet_model, 
        path="unet_resnet50.pth",
        classes=1
    )
    
    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.dicom_ad_cnn,
            model_fn=model_fn
        ),
        prep_fn=prepare_dicom_cnn,
        compressor_fn=compression.compress_cnn_dicom_ad,
        metrics_fn=compute_dicom_quality_metrics,
        bpp=16,
    )

# --- Autoencoder ---
# - pgm - pixel prediction autoencoder
def test_preprocessing_autoencoder(source_regex: re.Pattern):
    loader, _ = dloader.get_loader("../datasets/pgm/test")
    model = get_custom_model("autoencoder.pth")

    model_fn = partial(
        get_custom_model,
        path="autoencoder.pth",
    )

    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.ad_autoencoder,
            model_fn=model_fn,
        ),
        prep_fn=prepare_cnn,
        compressor_fn=compression.compress_cnn_ad,
        metrics_fn=compute_quality_metrics,
        bpp=8,
    )

# - pgm - ref pixels compression autoencoder
def test_preprocessing_autoencoder_compression(source_regex: re.Pattern):
    loader, _ = dloader.get_loader("../datasets/pgm/test")
    K = 5

    run_preprocessing(
        loader,
        predictor_fn=partial(
            prediction.ad_unfold_ridge_border,
            K=K
        ),
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_quality_metrics,
        bpp=8,
        autoencoder=True
    )

# ----- All pipeline tests -----

# --- Classic Ridge ---
# - pgm -
def test_ad_unfold_ridge_border(show=False):
    loader, _ = dloader.get_loader("../datasets/pgm/test")
    K = 5

    run_experiment(
        loader,
        predictor_fn=partial(
            prediction.ad_unfold_ridge_border,
            K=K
        ),
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_quality_metrics,
        receiver_fn=partial(
            receiver.receive,
            K=K
        ),
        bpp=8,
        show=show,
    )

# - dcm -
def test_dicom_ad_unfold_ridge_border(show=False):
    loader, _ = dloader.get_dicom_loader("../datasets/dcm/test")
    K = 5

    run_experiment(
        loader,
        predictor_fn=partial(
            prediction.dicom_ad_unfold_ridge_border,
            K=K
        ),
        prep_fn=prepare_dicom,
        compressor_fn=compression.compress_dicom_ad,
        metrics_fn=compute_dicom_quality_metrics,
        receiver_fn=partial(
            receiver.receive_dicom,
            K=K
        ),
        bpp=16,
        show=show,
    )

# --- CNN features + Ridge ---
# - pgm -
def test_ad_unet_mobilenet_v2_ridge(show=False):
    loader, _ = dloader.get_loader("../datasets/pgm/test", batch_size=1) # Keep batch size as 1 to avoid discrepancy
    K = 5
    model = get_mobilenet_v2_unet_model(classes=K ** 2)
    model.eval()
    save_model(model, "unet_mobilenetv2.pth") # save so it's the same for recovery

    model_fn = partial(
        get_mobilenet_v2_unet_model,
        path="unet_mobilenetv2.pth",
        classes=K ** 2
    )
    
    run_experiment(
        loader,
        predictor_fn=partial(
            prediction.ad_cnn_ridge,
            model_fn=model_fn
        ),
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_quality_metrics,
        receiver_fn=partial(
            receiver.receive_cnn_features, 
            model_fn=model_fn
        ),
        bpp=8,
        show=show,
    )

def test_ad_unet_resnet50_ridge(show=False):
    loader, _ = dloader.get_loader("../datasets/pgm/test", batch_size=1) # Keep batch size as 1 to avoid discrepancy
    K = 5
    model = get_resnet_50_unet_model(classes=K ** 2)
    model.eval()
    save_model(model, "unet_resnet50.pth") # save so it's the same for recovery

    model_fn=partial(
        get_resnet_50_unet_model, 
        path="unet_resnet50.pth",
        classes=K ** 2
    )
    
    run_experiment(
        loader,
        predictor_fn=partial(
            prediction.ad_cnn_ridge,
            model_fn=model_fn
        ),
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_quality_metrics,
         receiver_fn=partial(
            receiver.receive_cnn_features, 
            model_fn=model_fn
        ),
        bpp=8,
        show=show,
    )

# - dcm -

# --- Full CNN ---
# - pgm -
def test_ad_cnn(show=False):
    loader, _ = dloader.get_loader("../datasets/pgm/test")

    model_fn=partial(
        get_custom_model,
        path="cnn.pth",
    )

    run_experiment(
        loader,
        predictor_fn=partial(
            prediction.ad_cnn,
            model_fn=model_fn
        ),
        prep_fn=prepare_cnn,
        compressor_fn=compression.compress_cnn_ad,
        metrics_fn=compute_quality_metrics,
        receiver_fn=partial(
            receiver.receive_cnn, 
            model_fn=model_fn
        ),
        bpp=8,
        show=show,
    )

def test_ad_mobilenet_v2(show=False):
    loader, _ = dloader.get_loader("../datasets/pgm/test")
    model = get_mobilenet_v2_unet_model(classes=1)
    model.eval()
    save_model(model, "unet_mobilenetv2.pth") # save so it's the same for recovery

    model_fn=partial(
        get_mobilenet_v2_unet_model, 
        path="unet_mobilenetv2.pth",
        classes=1
    )
    
    run_experiment(
        loader,
        predictor_fn=partial(
            prediction.ad_cnn,
            model_fn=model_fn
        ),
        prep_fn=prepare_cnn,
        compressor_fn=compression.compress_cnn_ad,
        metrics_fn=compute_quality_metrics,
        receiver_fn=partial(
            receiver.receive_cnn, 
            model_fn=model_fn
        ),
        bpp=8,
        show=show,
    )

def test_ad_resnet_50(show=False):
    loader, _ = dloader.get_loader("../datasets/pgm/test")
    model = get_resnet_50_unet_model(classes=1)
    model.eval()
    save_model(model, "unet_resnet50.pth") # save so it's the same for recovery

    model_fn=partial(
        get_resnet_50_unet_model, 
        path="unet_resnet50.pth",
        classes=1
    )

    run_experiment(
        loader,
        predictor_fn=partial(
            prediction.ad_cnn,
            model_fn=model_fn
        ),
        prep_fn=prepare_cnn,
        compressor_fn=compression.compress_cnn_ad,
        metrics_fn=compute_quality_metrics,
        receiver_fn=partial(
            receiver.receive_cnn, 
            model_fn=model_fn
        ),
        bpp=8,
        show=show,
    )

# - dcm -

import torch
from torch.utils.data import DataLoader
from collections.abc import Callable

import backend.data.show as dshow
import backend.data.loader as dloader
import backend.predictor.predict as prediction
import backend.compressor.compress as compression
import backend.compressor.encryption as encryption
import backend.compressor.hiding as hiding
import backend.receiver.receive as receiver

from backend.models.models import *
from backend.predictor.results import Prediction, compute_dicom_metrics, compute_metrics
from backend.predictor.pipelines import reference_mask ## <--- bad should be elevated higher not owned by the predictor


def test_ad_unfold_ridge_border(show=False):
    loader, _ = dloader.get_loader("../datasets/BOSSbase_512")
    K = 5

    run_linear(
        loader,
        predictor_fn=prediction.ad_unfold_ridge_border,
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_metrics,
        receiver_fn=receiver.receive,
        K=K,
        bpp=8,
        dtype="uint8",
        show=show,
    )


def test_dicom_ad_unfold_ridge_border(show=False):
    loader, _ = dloader.get_dicom_loader("../datasets/dicom library 300")
    K = 5

    run_linear(
        loader,
        predictor_fn=prediction.dicom_ad_unfold_ridge_border,
        prep_fn=prepare_dicom,
        compressor_fn=compression.compress_dicom_ad,
        metrics_fn=compute_dicom_metrics,
        receiver_fn=receiver.receive_dicom,
        bpp=16,
        K=K,
        dtype="int16",
        show=show,
    )


def test_ad_mobilenet_ridge_border(show=False):
    loader, _ = dloader.get_loader("../datasets/BOSSbase_512", batch_size=1) # Keep batch size as 1 to avoid discrepancy
    K = 5
    model = get_torch_unet_model(classes=K ** 2)
    model.eval()
    save_model(model, "unet_mobilenetv2.pth") # save so it's the same for recovery
    
    run_linear(
        loader,
        predictor_fn=prediction.ad_mobilenetv2_ridge,
        prep_fn=prepare_pgm,
        compressor_fn=compression.compress_pgm_ad,
        metrics_fn=compute_metrics,
        receiver_fn=receiver.receive_cnn_features,
        bpp=8,
        K=K,
        dtype="uint8",
        show=show,
    )


#---- Generic runner ----

def run_linear(loader: DataLoader, predictor_fn: Callable, prep_fn: Callable, compressor_fn: Callable, 
               receiver_fn: Callable, metrics_fn: Callable, K: int, bpp: int, dtype: str, show: bool = False):
    """Runs experiment with linear predictor

    Args:
        loader (DataLoader): data loader for the image dataset
        predictor_fn (Callable): function for the prediction
        prep_fn (Callable): function for retrieving AD args
        compressor_fn (Callable): function for the compression
        receiver_fn (Callable): function for receiving
        metric_fn (Callable): function for metrics calculation
        bpp (int): image bpp
        dtype (str): image pixel dtype
        show (bool, optional): option to show the images on each step - defaults to False
    """
    def avg_er(new_er: float) -> float:
        avg_er.img_no += 1
        avg_er.rates += new_er
        return avg_er.rates / avg_er.img_no
    avg_er.img_no = 0; avg_er.rates = 0

    torch.manual_seed(0)
    
    K_e = "password1"
    K_h = "password2"
    message = "bardzo tajna wiadomosc"

    for batch in loader:
        H, W = batch.shape[-2:]
        mask = reference_mask(H, W)

        for raw_ad in predictor_fn(batch, K):
            ad_args, metric_args, original = prep_fn(raw_ad)

            ad = compressor_fn((H, W), *ad_args)

            prediction = Prediction(
                ad=ad,
                bpp=bpp,
                shape=(H, W),
                metrics=metrics_fn(
                    **metric_args,
                    mask=mask,
                    ad_bits=len(ad),
                ),
            )

            encrypted = encryption.encrypt_ad(
                ad,
                prediction.pixels,
                prediction.bpp,
                K_e,
            )

            stego = hiding.hider(
                encrypted,
                prediction.metrics.payload_capacity,
                message,
                K_h,
            )

            reconstructed, message = receiver_fn(
                stego,
                K_e,
                K_h,
                prediction.shape,
                K,
            )

            original_bytes = (
                original.contiguous()
                .numpy()
                .astype(dtype)
                .tobytes()
            )

            dshow.check_images(
                original_bytes,
                reconstructed.tobytes(),
            )

            print(f"Hidden Message: {message}")
            print(f"ER: {prediction.metrics.embedding_rate}")
            print(f"PSNR: {prediction.metrics.psnr}")
            print(f"SSIM: {prediction.metrics.ssim}")
            print(f"Avg ER: {avg_er(prediction.metrics.embedding_rate)}")
            
            if show:
                dshow.show_image(original_bytes, title="Original Image")
                dshow.show_image(stego, title="Stego Image")
                dshow.show_image(reconstructed.tobytes(), title="Reconstructed Image")


#---- Functions for getting the AD for PGM/Dicom ----

def prepare_pgm(raw_ad):
    kernel_weights, ref_pixels, error_map, original = raw_ad

    return (
        (kernel_weights, ref_pixels, error_map),
        {"original": original, "error_map": error_map},
        original,
    )


def prepare_dicom(raw_ad):
    img1_error_map, kernel_weights, ref_pixels, img2_error_map, original = raw_ad

    return (
        (img1_error_map, kernel_weights, ref_pixels, img2_error_map),
        {"original": original, "lsb_error_map": img2_error_map},
        original,
    )


def main():
    #test_ad_unfold_ridge_border()
    #test_dicom_ad_unfold_ridge_border()
    test_ad_mobilenet_ridge_border()

if __name__ == "__main__":
    main()
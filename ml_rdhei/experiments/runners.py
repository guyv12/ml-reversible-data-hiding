import torch
from torch.utils.data import DataLoader
from collections.abc import Callable

import backend.data.show as dshow
import backend.compressor.encryption as encryption
import backend.compressor.hiding as hiding
import backend.receiver.test as test_receiver

from backend.predictor.results import Prediction
from backend.predictor.pipelines import reference_mask ## <--- bad should be elevated higher not owned by the predictor

from experiments.autoencoder import * ## <-- import fix


#---- Generic runner ----

def run_experiment(loader: DataLoader, predictor_fn: Callable,
               prep_fn: Callable, compressor_fn: Callable, 
               receiver_fn: Callable, metrics_fn: Callable,
               bpp: int,
               show: bool = False) -> None:
    """Runs experiment

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

        for raw_ad in predictor_fn(batch):
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

            reconstructed, message, errors = receiver_fn(
                stego_image=stego,
                key_ad=K_e,
                key_msg=K_h,
                img_size=prediction.shape,
            )

            if errors:
                raise errors[0]

            test_receiver.test_image_reconstruction(
                original,
                reconstructed
            )

            print(f"Hidden Message: {message}")
            print(f"ER: {prediction.metrics.embedding_rate}")
            print(f"PSNR: {prediction.metrics.psnr}")
            print(f"SSIM: {prediction.metrics.ssim}")
            print(f"Avg ER: {avg_er(prediction.metrics.embedding_rate)}")
            
            if show:
                dshow.show_image(original, title="Original Image")
                dshow.show_bytes(stego, title="Stego Image")
                dshow.show_image(reconstructed, title="Reconstructed Image")


def run_preprocessing(loader: DataLoader, predictor_fn: Callable, 
                      prep_fn: Callable, compressor_fn: Callable,
                      metrics_fn: Callable, bpp: int, autoencoder: bool = False) -> None:
    """Runs preprocessing for the experiment

    Args:
        loader (DataLoader): data loader for the image dataset
        predictor_fn (Callable): function for the prediction
        prep_fn (Callable): function for retrieving AD args
        compressor_fn (Callable): function for the compression
    """
    def avg_er(new_er: float) -> float:
        avg_er.img_no += 1
        avg_er.rates += new_er
        return avg_er.rates / avg_er.img_no
    avg_er.img_no = 0; avg_er.rates = 0

    torch.manual_seed(0)

    for batch in loader:
        H, W = batch.shape[-2:]
        mask = reference_mask(H, W)

        for raw_ad in predictor_fn(batch):
            ad_args, metric_args, _ = prep_fn(raw_ad)

            ad = compressor_fn((H, W), *ad_args, autoencoder=autoencoder)

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

            print(f"AD Length: {len(ad) // 8} bytes")
            print(f"ER: {prediction.metrics.embedding_rate:.3f}")
            print(f"PSNR: {prediction.metrics.psnr:.3f}")
            print(f"SSIM: {prediction.metrics.ssim:.3f}")
            print(f"Avg ER: {avg_er(prediction.metrics.embedding_rate):.3f}")


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

def prepare_cnn(raw_ad):
    ref_pixels, error_map, original = raw_ad

    return (
        (ref_pixels, error_map),
        {"original": original, "error_map": error_map},
        original,
    )

def prepare_dicom_cnn(raw_ad):
    img1_error_map, img2_ref_pixels, img2_error_map, original = raw_ad

    return (
        (img1_error_map, img2_ref_pixels, img2_error_map),
        {"original": original, "lsb_error_map": img2_error_map},
        original,
    )

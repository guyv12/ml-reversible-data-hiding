import numpy as np

import backend.data.loader as dloader
import backend.predictor.predict as predict
import backend.compressor.compress as compress

from backend.predictor.pipelines import reference_mask
import backend.predictor.results as results

import backend.compressor.encryption as encryption
from backend.data.show import show_image, check_images
from backend.receiver.receive import receive, receive_dicom
from backend.compressor.hiding import hider


def test_ad_unfold_ridge_border(show: bool = False):
    loader, _ = dloader.get_loader("../datasets/BOSSbase_512")
    H, W = 512, 512 # All images in BOSSbase are 512x512
    mask = reference_mask(H, W) ### predictor shouldn't own mask it should be elevated...

    K_e = "password1"
    K_h = "password2"

    for i, batch in enumerate(loader):
        for raw_ad in predict.ad_unfold_ridge_border(batch, K=5):

            # 1. Get AD
            kernel_weights, ref_pixels, error_map, original_image = raw_ad

            # 2. Compress AD string
            ad = compress.compress_pgm_ad(
                (H, W), kernel_weights, ref_pixels,error_map
            )
            
            ### Get prediction metrics ### 
            prediction_metrics = results.Prediction(
                ad=ad,
                bpp=8,
                shape=(H, W),
                metrics=results.compute_metrics(
                    original=original_image, error_map=error_map,
                    mask=mask, ad_bits=len(ad),
                )
            )

            # 3. Encrypt AD
            encrypted_ad = encryption.encrypt_ad(
                ad, prediction_metrics.pixels, prediction_metrics.bpp, K_e
            )

            # 4. Hide AD in the image
            stego_image = hider(
                encrypted_ad, prediction_metrics.metrics.payload_capacity, "bardzo tajna wiadomosc", K_h
            )

            # 5. Reconstruct the image based on decrypted AD
            reconstructed_image, message = receive(
                stego_image, K_e, K_h, prediction_metrics.shape
            )
            
            reconstructed_image = reconstructed_image.tobytes()

            # 6. Verify the reconstruction is successful
            original_bytes = (
                original_image.contiguous().numpy().astype("uint8").tobytes()
            )
            check_images(original_bytes, reconstructed_image)
            print(message)
            print(prediction_metrics.metrics.embedding_rate)
            print(prediction_metrics.metrics.psnr)
            print(prediction_metrics.metrics.ssim)
            print()

            if show:
                show_image(original_bytes, title="Original Image")
                show_image(stego_image, title="Stego Image")
                show_image(reconstructed_image, title="Reconstructed Image")


def test_dicom_ad_unfold_ridge_border(show: bool = False):
    DICOM_loader, _ = dloader.get_dicom_loader("../datasets/dicom library 300")

    ########################################### to delete if dicom metrics in place
    bpp = 16
    def pixels(H, W):
        return H * W
    def payload_capacity(pixels, ad_bits):
        return (pixels * bpp) - ad_bits
    def embedding_rate(capacity, pixels):
        return capacity / pixels
    ###########################################

    K_e = "password1"
    K_h = "password2"

    for i, batch in enumerate(DICOM_loader):
        H, W = batch.shape[-2:]
        mask = reference_mask(H, W)

        for raw_ad in predict.dicom_ad_unfold_ridge_border(batch):

            # 1. Get AD
            img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map, original_image = raw_ad

            # 2. Compress AD string
            ad = compress.compress_dicom_ad(
                (H, W), img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map
            )

            # TODO Dicom Prediction Metrics?

            # 3. Encrypt AD
            encrypted_ad = encryption.encrypt_ad(
                ad, pixels(H, W), bpp, K_e
            )

            # 4. Hide AD in the image
            stego_image = hider(
                encrypted_ad, payload_capacity(pixels(H, W), len(ad)), "bardzo tajna wiadomosc", K_h
            )

            # 5. Reconstruct the image based on decrypted AD
            reconstructed_image, message = receive_dicom(
                stego_image, K_e, K_h, (H, W)
            )
            
            reconstructed_image = reconstructed_image.tobytes()

            # 6. Verify the reconstruction is successful
            original_bytes = (
                original_image.contiguous().numpy().astype("int16").tobytes()
            )
            check_images(original_bytes, reconstructed_image)
            print(message)
            print(embedding_rate(payload_capacity(pixels(H, W), len(ad)), pixels(H, W)))
            print()

            if show:
                show_image(original_bytes, dtype=np.int16, width=W, height=H, title="Original Image")
                # show_image(stego_image, width=W, height=H, title="Stego Image")
                show_image(reconstructed_image, dtype=np.int16, width=W, height=H, title="Reconstructed Image")


# test_ad_unfold_ridge_border(False)
test_dicom_ad_unfold_ridge_border(False)
from data.test import test_dataloader_time
from ml_rdhei.compressor.hiding import hider
from ml_rdhei.predictor.test import test_sklearn_kernel

import data.loader as dloader
import predictor.predict as ppredict
import compressor.compress as ccompress
import compressor.encryption as encryption


def pgm_main():
    ## sklearn model -- we use yield in torch there would be 1 loop
    BOSSBase_loader, _ = dloader.get_loader("datasets/BOSSbase_512")

    rates = 0
    counter = 0
    pixels = 512 * 512
    bpp = 8
    bits_per_image = pixels * bpp
    K_e = "password"
    K_h = "password"

    for i, batch in enumerate(BOSSBase_loader):
        for raw_ad in ppredict.pgm_raw_ad_sklearn(batch):
            kernel_weights, ref_pixels, error_map = raw_ad
            ad = ccompress.compress_pgm_ad((512, 512), kernel_weights, ref_pixels, error_map)
            ad_enrypted = encryption.encrypt_data(ad, K_e)

            available_bits = bits_per_image - (len(ad) * bpp)
            emb_rate = available_bits / pixels
            rates += emb_rate
            counter += 1
            print(f"Batch:{i} | Ad Length: {len(ad)}")
            print(f"Current embedding rate[bpp]: {emb_rate:.4f}")
            print(f"Avg embedding rate[bpp]: {rates/counter:.4f}\n")

            image = hider(ad_enrypted, available_bits//8, "bardzo tajna wiadomosc", K_h)

    return

def dicom_main():
    DICOM_loader, _ = dloader.get_loader("datasets/BOSSbase_512")

    rates = 0
    counter = 0
    pixels = 512 * 512
    bpp = 16
    bits_per_image = pixels * bpp

    for i, batch in enumerate(DICOM_loader):
        for raw_ad in ppredict.dicom_raw_ad_sklearn(batch):
            img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map = raw_ad
            
            ad = ccompress.compress_dicom_ad((512, 512), img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map)

            available_bits = bits_per_image - (len(ad) * bpp)
            emb_rate = available_bits / pixels
            rates += emb_rate
            counter += 1
            print(f"Batch:{i} | Ad Length: {len(ad)}")
            print(f"Current embedding rate[bpp]: {emb_rate:.4f}")
            print(f"Avg embedding rate[bpp]: {rates/counter:.4f}\n")
    
    return

def main() -> None:
    pgm_main()
    # dicom_main()


if __name__ == "__main__":
    main()

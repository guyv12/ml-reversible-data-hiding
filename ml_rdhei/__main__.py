from data.test import test_dataloader_time
from ml_rdhei.predictor.test import test_sklearn_kernel

import data.loader as dloader
import predictor.predict as ppredict
import compressor.compress as ccompress


def pgm_main():
    
    ## sklearn model -- we use yield in torch there would be 1 loop
    BOSSBase_loader, _ = dloader.get_loader("datasets/BOSSbase_512")

    rates = 0
    counter = 0
    pixels = 512 * 512
    bits_per_image = pixels * 8

    for i, batch in enumerate(BOSSBase_loader):
        for raw_ad in ppredict.pgm_raw_ad_sklearn(batch):
            kernel_weights, ref_pixels, error_map = raw_ad
            ad = ccompress.compress_pgm_ad((512, 512), kernel_weights, ref_pixels, error_map)

            available_bits = bits_per_image - (len(ad)*8)
            emb_rate = available_bits / pixels
            rates += emb_rate
            counter += 1
            print(f"Batch:{i} | Ad Length: {len(ad)}")
            print(f"Current embedding rate[bpp]: {emb_rate:.4f}")
            print(f"Avg embedding rate[bpp]: {rates/counter:.4f}\n")

    return

def dicom_main():
    DICOM_loader, _ = dloader.get_loader("datasets/BOSSbase_512")

    rates = 0
    counter = 0
    pixels = 512 * 512
    bits_per_image = pixels * 16

    for i, batch in enumerate(DICOM_loader):
        for raw_ad in ppredict.dicom_raw_ad_sklearn(batch):
            img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map = raw_ad
            
            # ad = ccompress.compress_ad_dicom(kernel_weights, ref_pixels, error_map, batch)

            # available_bits = bits_per_image - (len(ad)*8)
            # emb_rate = available_bits / pixels
            # rates += emb_rate
            # counter += 1
            # print(f"Batch:{i} | Ad Length: {len(ad)}")
            # print(f"Current embedding rate[bpp]: {emb_rate:.4f}")
            # print(f"Avg embedding rate[bpp]: {rates/counter:.4f}\n")

def main() -> None:
    # test_dataloader_time()
    # test_sklearn_kernel()
    pgm_main()


if __name__ == "__main__":
    main()

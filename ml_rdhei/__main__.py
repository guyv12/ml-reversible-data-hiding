import backend.data.loader as dloader
import backend.predictor.predict as ppredict
import backend.compressor.compress as ccompress
import backend.compressor.encryption as encryption
from backend.predictor.predict import ADRidge, ADRidgeDicom
from backend.data.show import show_image, check_images
from backend.receiver.receive import receive
from backend.compressor.hiding import hider


def pgm_main():
    ## sklearn model -- we use yield in torch there would be 1 loop
    BOSSBase_loader, _ = dloader.get_loader("../datasets/BOSSbase_512")

    rates = 0
    counter = 0
    pixels = 512 * 512
    bpp = 8
    bits_per_image = pixels * bpp
    K_e = "password"
    K_h = "password"

    for i, batch in enumerate(BOSSBase_loader):
        for raw_ad in ADRidge.get_ad(batch):
            kernel_weights, ref_pixels, error_map, original = raw_ad
            original_bytes = original.contiguous().cpu().numpy().astype('uint8').tobytes()
            show_image(original_bytes)

            ad = ccompress.compress_pgm_ad((512, 512), kernel_weights, ref_pixels, error_map)
            ad_enrypted = encryption.encrypt_ad(ad, pixels, bpp, K_e)

            available_bits = bits_per_image - len(ad)
            emb_rate = available_bits / pixels
            rates += emb_rate
            counter += 1
            print(f"Batch:{i} | Ad Length: {len(ad)}")
            print(f"Current embedding rate[bpp]: {emb_rate:.4f}")
            print(f"Avg embedding rate[bpp]: {rates/counter:.4f}\n")

            image = hider(ad_enrypted, available_bits//8, "bardzo tajna wiadomosc", K_h)
            show_image(image)
            reconstructed = receive(image, K_e, K_h, len(ref_pixels)).tobytes()
            check_images(original_bytes, reconstructed)
            show_image(reconstructed)

    return

def dicom_main():
    DICOM_loader, _ = dloader.get_dicom_loader("../datasets/DICOM")

    rates = 0
    counter = 0
    pixels = 512 * 512
    bpp = 16
    bits_per_image = pixels * bpp

    for i, batch in enumerate(DICOM_loader):
        H, W = batch.shape[-2:]

        for raw_ad in ADRidgeDicom.get_ad(batch):
            img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map = raw_ad
            
            ad = ccompress.compress_dicom_ad((H, W), img1_error_map, img2_kernel_weights, img2_ref_pixels, img2_error_map)

            pixels = H * W
            bits_per_image = pixels * bpp

            available_bits = bits_per_image - (len(ad) * 8)
            emb_rate = available_bits / pixels
            rates += emb_rate
            counter += 1
            print(f"Batch:{i} | Ad Length: {len(ad)}")
            print(f"Current embedding rate[bpp]: {emb_rate:.4f}")
            print(f"Avg embedding rate[bpp]: {rates/counter:.4f}\n")
    
    return

def main() -> None:
    pgm_main()
    #dicom_main()


if __name__ == "__main__":
    main()

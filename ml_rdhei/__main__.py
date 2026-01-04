from data.test import test_dataloader_time
from ml_rdhei.predictor.test import test_sklearn_kernel

import data
import predictor.models as pmodels
import predictor.predict as ppredict
import compressor.compress as ccompress


def pref_main():
    # get image
    img = None
    model = pmodels.get_sklearn_model()

    kernel_weights, ref_pixels, error_map = ppredict.get_raw_ad_sklearn(img, model)
    ad = ccompress.compress_ad_classic(kernel_weights, ref_pixels, error_map)

    # embed
    return

def main() -> None:
    #test_dataloader_time()
    test_sklearn_kernel()


if __name__ == "__main__":
    main()

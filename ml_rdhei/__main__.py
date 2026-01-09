from data.test import test_dataloader_time
from ml_rdhei.predictor.test import test_sklearn_kernel

import data.loader as dloader
import predictor.models as pmodels
import predictor.predict as ppredict
import compressor.compress as ccompress


def pref_main():
    
    ## sklearn model -- we use yield in torch there would be 1 loop
    BOSSBase_loader, _ = dloader.get_loader("datasets/BOSSbase_512")
    model = pmodels.get_sklearn_model()

    for i, batch in enumerate(BOSSBase_loader):
        for raw_ad in ppredict.get_raw_ad_sklearn(batch, model):
            kernel_weights, ref_pixels, error_map = raw_ad
            ad = ccompress.compress_ad_classic(kernel_weights, ref_pixels, error_map)

            print(f"Batch:{i} | Ad Length: {len(ad)}")

    return


def main() -> None:
    #test_dataloader_time()
    #test_sklearn_kernel()
    pref_main()


if __name__ == "__main__":
    main()

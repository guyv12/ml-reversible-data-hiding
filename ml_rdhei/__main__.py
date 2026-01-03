from data.test import test_dataloader_time
from ml_rdhei.predictor.test import test_sklearn_kernel

import data
import predictor.models as pmodels
import predictor.predict as ppredict
import compressor.compress as ccompress


def pref_main():
    # get image
    model = pmodels.get_sklearn_model()

    stuff = ppredict.get_raw_ad()
    ad = ccompress.compress_ad(stuff)

    # embed

def main() -> None:
   test_sklearn_kernel()


if __name__ == "__main__":
    main()

import re
import argparse

from experiments.pipelines import *


# ---------- RUN ------------

PIPELINE_EXPERIMENTS = {
    "pgm_ridge": test_ad_unfold_ridge_border,
    "dicom_ridge": test_dicom_ad_unfold_ridge_border,
    "pgm_mobilenet_ridge": test_ad_unet_mobilenet_v2_ridge,
    "pgm_resnet_ridge": test_ad_unet_resnet50_ridge,
    #dicom_mobilenet_ridge, -- not needed
    #dicom_resnet_ridge, -- not needed
    "pgm_custom_cnn": test_ad_cnn,
    "pgm_mobilenet_cnn": test_ad_mobilenet_v2,
    "pgm_resnet_cnn": test_ad_resnet_50,
    #dicom_custom_cnn, -- not needed
    #dicom_mobilenet_cnn, --not needed
    #dicom_resnet_cnn --not needed
}

PREPROCESSING_EXPERIMENTS = {
    "pgm_ridge": test_preprocessing_ad_unfold_ridge_border,
    "dicom_ridge": test_preprocessing_dicom_ad_unfold_ridge_border,
    #pgm_mobilenet_ridge,
    #pgm_resnet_ridge,
    "dicom_mobilenet_ridge": test_preprocessing_dicom_ad_unet_mobilenet_v2_ridge,
    #dicom_resnet_ridge,
    "pgm_custom_cnn": test_preprocessing_ad_cnn,
    #pgm_mobilenet_cnn
    #pgm_resnet_cnn
    #dicom_custom_cnn,
    "dicom_mobilenet_cnn": test_preprocessing_dicom_ad_mobilenet_v2,
    #dicom_resnet_cnn
}

def get_source_regex(source: str):
    if source == "all":
        return None

    return re.compile(rf"^{re.escape(source)}", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description="Run pipeline/experiment")

    parser.add_argument("--mode", choices=["pipeline", "preprocessing"], default="preprocessing", help="Experiment mode")
    parser.add_argument("--experiment", required=True, help="Name of the experiment to run")
    parser.add_argument("--source", choices=["boss", "bows", "car", "tcia", "all"], default="all")
    parser.add_argument("--show", action="store_true", help="Display results during a full pipeline experiment")
    args = parser.parse_args()

    experiments = (PIPELINE_EXPERIMENTS if args.mode == "pipeline"
        else PREPROCESSING_EXPERIMENTS
    )
    source_regex = get_source_regex(args.source)

    if args.experiment not in experiments:
        parser.error(f"Experiment '{args.experiment}' is not available for mode '{args.mode}'")

    runner = experiments[args.experiment]

    print(f"Mode:       {args.mode}")
    print(f"Experiment: {args.experiment}")

    if args.mode == "pipeline":
        runner(show=args.show)
    else:
        runner(source_regex=source_regex)


if __name__ == "__main__":
    main()

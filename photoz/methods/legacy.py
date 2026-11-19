from pathlib import Path


def predict_capsnet(config_path: str, **_: object) -> None:
    from photoz.methods.capsnet import run_inference

    run_inference(Path(config_path))


def predict_clip(config_path: str, **_: object) -> None:
    from photoz.methods.euclid_clip import run_inference as run

    run(Path(config_path))


NAMED_PREDICTORS = {"capsnet": predict_capsnet, "euclid_clip": predict_clip}

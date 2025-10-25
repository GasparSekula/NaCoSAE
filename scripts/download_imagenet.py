from collections.abc import Hashable
from collections.abc import Iterable
from collections.abc import Sequence
import immutabledict
import os

from absl import app
from absl import flags
from absl import logging
import pandas as pd

import classes

_DOWNLOAD_DIR = flags.DEFINE_string(
    "download_dir", "download_tmp", "Temp directory for parquet files."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "imagenet", "Output directory."  # so descriptive help lol
)
_CLEANUP = flags.DEFINE_bool(
    "cleanup",
    True,
    "If True parquet files will be deleted after processing.",
)

_VALIDATION_PARQUET_FILES = tuple(
    f"data/validation-000{idx if idx >= 10 else str(0) + str(idx)}-of-00014.parquet"
    for idx in range(14)
)
_LABEL_NAMES_SPECIAL_CASES = immutabledict.immutabledict(
    {
        "maillot, tank suit": "tank_suit",
        "Cardigan, Cardigan Welsh corgi": "cardigan_welsh_corgi",
    }
)


def _download_parquet_files() -> None:
    """Downloads parquet files from huggingface."""
    os.makedirs(_DOWNLOAD_DIR.value, exist_ok=True)
    hf_token = os.environ["HF_TOKEN"]

    hf_command = f"""
    hf download imagenet-1k \
    --repo-type dataset \
    --local-dir {_DOWNLOAD_DIR.value} \
    --token {hf_token} \
    --include {" ".join(_VALIDATION_PARQUET_FILES)} \
    --quiet
    """.strip()

    logging.info("Downloading parquet files from huggingface.")
    os.system(hf_command)


def _process_label_name(label_name: str) -> str:
    """Processes label name by taking the first synonym, replacing spaces and converting to lower case."""
    if label_name in _LABEL_NAMES_SPECIAL_CASES:
        return _LABEL_NAMES_SPECIAL_CASES[label_name]
    return label_name.split(",")[0].replace(" ", "_").lower()


def _create_class_subdirectories() -> None:
    """Creates subdirectories for each class."""
    logging.info("Creating class subdirectories.")
    os.mkdir(_OUTPUT_DIR.value)
    for label_name in classes.IMAGENET2012_CLASSES.values():
        dir_name = _process_label_name(label_name)
        try:
            os.mkdir(f"{_OUTPUT_DIR.value}/{dir_name}")
        except FileExistsError:
            logging.warning("Repeated class: %s" % dir_name)


def _save_images(
    labels_list: Sequence[str], rows: Iterable[tuple[Hashable, pd.Series]]
) -> None:
    """Saves images from rows of image DataFrame to corresponding class directory."""
    for _, (image, label) in rows:
        label_name = labels_list[label]
        dir_name = _process_label_name(label_name)
        image_bytes = image["bytes"]
        filename = image["path"]
        with open(f"{_OUTPUT_DIR.value}/{dir_name}/{filename}", "wb") as f:
            f.write(image_bytes)


def _process_parquet_files() -> None:
    """Reads parquet files and saves images from image DataFrames."""
    dir_path = f"{_DOWNLOAD_DIR.value}/data"
    labels_list = list(classes.IMAGENET2012_CLASSES.values())

    for filename in os.listdir(dir_path):
        logging.info("Processing file: %s" % filename)
        image_dataframe = pd.read_parquet(f"{dir_path}/{filename}")
        _save_images(labels_list, image_dataframe.iterrows())

    logging.info("All parquet files processed.")


def _cleanup() -> None:
    """Deleting parquet files and temp directory."""
    logging.info("Deleting parquet files.")
    for validation_parquet_file in _VALIDATION_PARQUET_FILES:
        os.system(f"rm {_DOWNLOAD_DIR.value}/{validation_parquet_file}")
    logging.info("Deleting temp directory.")
    os.system(f"rm -r {_DOWNLOAD_DIR.value}")


def main(argv):
    _download_parquet_files()
    _create_class_subdirectories()
    _process_parquet_files()
    if _CLEANUP.value:
        _cleanup()


if __name__ == "__main__":
    app.run(main)

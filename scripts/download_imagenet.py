"""Download and organize ImageNet images from Hugging Face.

This script downloads ImageNet validation parquet files from Hugging Face,
processes them, and saves images organized into subdirectories by class label.
The label names are processed to create valid directory names.
"""

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
    """Download parquet files from Hugging Face using the HF_TOKEN environment variable.

    Creates the download directory if it doesn't exist and downloads validation
    parquet files for ImageNet-1k dataset.
    """
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
    """Process ImageNet label name into a valid directory name.

    Takes the first synonym from the label, replaces spaces with underscores,
    and converts to lowercase. Handles special cases with multiple similar labels.

    Args:
        label_name: The original ImageNet label name.

    Returns:
        Processed label name suitable for use as a directory name.
    """
    if label_name in _LABEL_NAMES_SPECIAL_CASES:
        return _LABEL_NAMES_SPECIAL_CASES[label_name]
    return label_name.split(",")[0].replace(" ", "_").lower()


def _create_class_subdirectories() -> None:
    """Create subdirectories for each ImageNet class.

    Creates the output directory and a subdirectory for each of the 1000
    ImageNet classes with processed label names.
    """
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
    """Save images from DataFrame rows to their corresponding class directories.

    Iterates through image rows, extracts image bytes and filenames, and
    writes them to the appropriate class subdirectory.

    Args:
        labels_list: Sequence of class label names indexed by class ID.
        rows: Iterable of tuples containing image data and label index.
    """
    for _, (image, label) in rows:
        label_name = labels_list[label]
        dir_name = _process_label_name(label_name)
        image_bytes = image["bytes"]
        filename = image["path"]
        with open(f"{_OUTPUT_DIR.value}/{dir_name}/{filename}", "wb") as f:
            f.write(image_bytes)


def _process_parquet_files() -> None:
    """Read parquet files and save images to class directories.

    Iterates through all parquet files in the download directory, reads each
    one as a DataFrame, and saves the images to their corresponding class
    subdirectories.
    """
    dir_path = f"{_DOWNLOAD_DIR.value}/data"
    labels_list = list(classes.IMAGENET2012_CLASSES.values())

    for filename in os.listdir(dir_path):
        logging.info("Processing file: %s" % filename)
        image_dataframe = pd.read_parquet(f"{dir_path}/{filename}")
        _save_images(labels_list, image_dataframe.iterrows())

    logging.info("All parquet files processed.")


def _cleanup() -> None:
    """Delete downloaded parquet files and temporary download directory.

    Removes all validation parquet files and deletes the entire temporary
    download directory to free up disk space.
    """
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

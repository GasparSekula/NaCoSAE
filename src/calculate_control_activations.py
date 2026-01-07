"""Calculate and save control images activations."""

from collections.abc import Iterator
import os
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
from PIL import Image
import torch
from tqdm import tqdm

import image_processing
from model import explained_model

_CONTROL_IMAGES_DIRECTORY = flags.DEFINE_string(
    "control_dir", "control_images", "Directory with control images."
)

_SAVE_PATH = flags.DEFINE_string(
    "save_path", "control_activations", "Path to save calculated activations."
)

_EXPLAINED_MODEL_ID = flags.DEFINE_string(
    "model_id", "resnet18", "ID of explained model."
)

_LAYER = flags.DEFINE_string(
    "layer", "avgpool", "Layer to collect actiavtions from."
)

_DEVICE = flags.DEFINE_enum(
    "device", "cuda", ("cuda", "cpu", "mps"), "Device to use."
)

flags.register_validator(
    "control_dir",
    lambda value: os.path.exists(value),
    "Provided control_dir path does not exist.",
)


def _get_concept_directories_and_names(
    control_images_directory: str,
) -> Iterator[Tuple[str, str]]:
    for concept_name in os.listdir(control_images_directory):
        concept_directory = os.path.join(control_images_directory, concept_name)
        yield concept_directory, concept_name


def _transform_concept_images(concept_directory: str) -> torch.Tensor:
    """
    Loads and transforms control images.
    """
    logging.info("Transforming images.")
    control_images = []
    for image_filename in os.listdir(concept_directory):
        image_path = os.path.join(concept_directory, image_filename)
        with Image.open(image_path) as pil_image:
            control_images.append(pil_image.copy())
    input_batch = image_processing.transform_images(control_images)
    return input_batch


def _calculate_concept_activations(
    explained_model: explained_model.ExplainedModel, input_batch: torch.Tensor
) -> torch.Tensor:
    return explained_model.get_activations(input_batch)


def _create_activations_save_path(
    save_path: str, model_id: str, layer: str
) -> str:
    """Creates save path for activations and returns directory name."""
    model_save_path = os.path.join(save_path, model_id, layer)
    logging.info("Creating path %s." % model_save_path)
    try:
        os.makedirs(model_save_path)
    except FileExistsError:
        logging.info(
            "Path %s already exists, skipping creation." % model_save_path
        )

    return model_save_path


def _save_concept_activations(
    save_path: str, concept_name: str, activations: torch.Tensor
) -> None:
    save_filepath = os.path.join(save_path, f"{concept_name}.pt")
    torch.save(activations, save_filepath)


def main(argv):
    logging.info("Loading model with model_id=%s." % _EXPLAINED_MODEL_ID.value)
    expl_model = explained_model.ExplainedModel(
        _EXPLAINED_MODEL_ID.value,
        _LAYER.value,
        _DEVICE.value,
        model_swapping=False,
    )
    model_save_path = _create_activations_save_path(
        _SAVE_PATH.value, _EXPLAINED_MODEL_ID.value, _LAYER.value
    )
    # TODO(piechotam) parallelise?
    for concept_directory, concept_name in tqdm(
        _get_concept_directories_and_names(_CONTROL_IMAGES_DIRECTORY.value)
    ):
        input_batch = _transform_concept_images(
            concept_directory,
        )
        activations = _calculate_concept_activations(expl_model, input_batch)
        _save_concept_activations(model_save_path, concept_name, activations)


if __name__ == "__main__":
    app.run(main)

"""Calculate and save control concept image activations.

This script loads control images for different concepts, processes them through
a specified neural network layer, and saves the resulting activation tensors
for use in concept scoring.
"""

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
    """Iterate over concept directories and their names.

    Args:
        control_images_directory: Path to directory containing concept subdirectories.

    Yields:
        Tuples of (directory_path, concept_name) for each concept.
    """
    for concept_name in os.listdir(control_images_directory):
        concept_directory = os.path.join(control_images_directory, concept_name)
        yield concept_directory, concept_name


def _transform_concept_images(concept_directory: str) -> torch.Tensor:
    """Load and preprocess control images from a concept directory.

    Loads all images from the specified directory, applies standard preprocessing
    and normalization transformations, and returns them as a batched tensor.

    Args:
        concept_directory: Path to directory containing control images for a concept.

    Returns:
        Batch tensor of transformed images ready for model inference.
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
    """Calculate layer activations for a batch of images.

    Args:
        explained_model: The explained model with a registered forward hook.
        input_batch: Batch of preprocessed images.

    Returns:
        Activation tensor from the registered layer.
    """
    return explained_model.get_activations(input_batch)


def _create_activations_save_path(
    save_path: str, model_id: str, layer: str
) -> str:
    """Create directory structure for saving activations.

    Creates nested directories for organizing activations by model and layer,
    creating them if they don't already exist.

    Args:
        save_path: Base save path for all activations.
        model_id: Model identifier (e.g., 'resnet18').
        layer: Layer name (e.g., 'avgpool').

    Returns:
        Full path to the created/existing model-layer directory.
    """
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
    """Save activation tensor for a concept to disk.

    Args:
        save_path: Directory to save the activation file in.
        concept_name: Name of the concept (becomes the filename without extension).
        activations: Activation tensor to save.
    """
    save_filepath = os.path.join(save_path, f"{concept_name}.pt")
    torch.save(activations, save_filepath)


def main(argv):
    """Main entry point for activation calculation pipeline.

    Loads a model with a registered forward hook, iterates through all control
    concepts, calculates their activations, and saves them to disk organized
    by model ID and layer.

    Args:
        argv: Command-line arguments (provided by absl).
    """
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

"""Calculate and save activations for each image separately."""

from typing import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from PIL import Image
import torch
from tqdm import tqdm

from src import image_processing
from model import explained_model


_CONCEPT_NAMES = flags.DEFINE_list(
    "concept_names",
    None,
    "List of concept names to process.",
    required=True,
)

_CONTROL_IMAGES_DIRECTORY = flags.DEFINE_string(
    "control_dir",
    "control_images",
    "Directory with control images.",
)

_EXPLAINED_MODEL_ID = flags.DEFINE_string(
    "model_id",
    "resnet18",
    "ID of explained model.",
)

_LAYER = flags.DEFINE_string(
    "layer",
    "avgpool",
    "Layer to collect activations from.",
)

_DEVICE = flags.DEFINE_enum(
    "device",
    "cuda",
    ("cuda", "cpu", "mps"),
    "Device to use.",
)

_SAVE_PATH = flags.DEFINE_string(
    "save_path",
    "analysis/results/control_activations",
    "Path to save calculated activations.",
)


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


def _process_single_image(
    image_path: str,
    expl_model: explained_model.ExplainedModel,
) -> torch.Tensor:
    """Load, transform, and get activations for a single image."""
    with Image.open(image_path) as pil_image:
        image = pil_image.copy()
    
    input_batch = image_processing.transform_images([image])
    
    activations = expl_model.get_activations(input_batch)
    
    return activations


def _save_image_activations(
    save_path: str,
    concept_name: str,
    image_filename: str,
    activations: torch.Tensor,
) -> None:
    """Save activations for a single image."""
    concept_save_path = os.path.join(save_path, concept_name)
    try:
        os.makedirs(concept_save_path)
    except FileExistsError:
        pass
    
    image_name = os.path.splitext(image_filename)[0]
    save_filepath = os.path.join(concept_save_path, f"{image_name}.pt")
    
    torch.save(activations, save_filepath)
    logging.debug(f"Saved activations to {save_filepath}")


def main(argv):
    """Calculate activations for each image separately and save with image identifier."""
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

    for concept_name in _CONCEPT_NAMES.value:
        concept_directory = os.path.join(
            _CONTROL_IMAGES_DIRECTORY.value, concept_name
        )
        
        if not os.path.exists(concept_directory):
            logging.warning(
                f"Concept directory '{concept_directory}' does not exist, skipping."
            )
            continue
        
        logging.info(f"Processing concept: {concept_name}")
        
        image_files = [
            f for f in os.listdir(concept_directory)
            if os.path.isfile(os.path.join(concept_directory, f))
            and not f.startswith('.')  
        ]
        
        for image_filename in tqdm(image_files, desc=f"Processing {concept_name}"):
            image_path = os.path.join(concept_directory, image_filename)
            
            try:
                activations = _process_single_image(image_path, expl_model)
                
                _save_image_activations(
                    model_save_path,
                    concept_name,
                    image_filename,
                    activations,
                )
            except Exception as e:
                logging.error(
                    f"Error processing image {image_filename} in concept "
                    f"{concept_name}: {str(e)}"
                )
                continue
        
        logging.info(
            f"Completed processing {len(image_files)} images for concept '{concept_name}'"
        )


if __name__ == "__main__":
    app.run(main)

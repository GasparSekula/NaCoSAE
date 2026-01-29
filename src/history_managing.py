"""Save pipeline history artifacts and results.

This module provides functions for saving and formatting pipeline artifacts
such as LLM generation history, generated images, and experiment parameters.
"""

import dataclasses
import os
import json
from typing import BinaryIO, Iterable, Sequence, TextIO, Mapping, Any

from PIL import Image

import config
import scoring


def _write_iterable_to_stream(
    output_stream: TextIO, lines: Iterable[str]
) -> None:
    """Write iterable contents to a text stream, one item per line.

    Args:
        output_stream: Text stream to write to.
        lines: Iterable of strings to write.
    """
    output_stream.write("\n".join(lines))


def save_llm_history(
    history: Iterable[str], save_directory: str, filename: str
) -> None:
    """Save language model generation history to a file.

    Args:
        history: Iterable of generation history strings.
        save_directory: Directory to save the history file to.
        filename: Name of the file to save.
    """
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, filename)

    with open(save_path, "w") as save_file:
        _write_iterable_to_stream(save_file, history)


def _write_image_to_stream(output_stream: BinaryIO, image: Image.Image) -> None:
    """Write an image to a binary stream in JPEG format.

    Args:
        output_stream: Binary stream to write image to.
        image: PIL Image object to save.
    """
    image.save(output_stream, format="jpeg")


def save_images_from_iteration(
    images: Sequence[Image.Image],
    save_directory: str,
    iter_number: int,
    concept: str,
) -> None:
    """Save images generated in a single pipeline iteration.

    Creates a subdirectory for the iteration and saves each image as a JPEG file.

    Args:
        images: Sequence of PIL Image objects to save.
        save_directory: Base directory to save images to.
        iter_number: Iteration number for organization.
        concept: Concept name, used in directory naming.
    """
    save_path = os.path.join(
        save_directory, f"iteration_{iter_number}_{concept.replace(" ", "_")}"
    )
    os.makedirs(save_path, exist_ok=True)
    for image_number, image in enumerate(images, 1):
        filename = f"image_{image_number}.jpeg"
        with open(os.path.join(save_path, filename), "wb") as save_file:
            _write_image_to_stream(save_file, image)


def save_pipeline_parameters(
    save_directory: str,
    run_id: str,
    load_config: config.LoadConfig,
    image_generation_config: config.ImageGenerationConfig,
    concept_history_config: config.ConceptHistoryConfig,
    history_managing_config: config.HistoryManagingConfig,
    neuron_id: int,
    metric: scoring.Metric,
    model_layer_activations_path: str,
):
    """Save all pipeline configuration parameters to a file.

    Creates a formatted text file with all experiment parameters including
    model configurations, generation settings, and neuron information.

    Args:
        save_directory: Directory to save parameters to.
        run_id: Unique identifier for this pipeline run.
        load_config: Model loading configuration.
        image_generation_config: Image generation settings.
        concept_history_config: Concept history initialization settings.
        history_managing_config: Result saving configuration.
        neuron_id: ID of the neuron being explained.
        metric: Activation metric being used.
        model_layer_activations_path: Path to control concept activations.
    """
    os.makedirs(save_directory, exist_ok=True)
    params = {
        "run_id": run_id,
        "load_config": dataclasses.asdict(load_config),
        "image_generation_config": dataclasses.asdict(image_generation_config),
        "concept_history_config": dataclasses.asdict(concept_history_config),
        "history_managing_config": dataclasses.asdict(history_managing_config),
        "neuron_id": neuron_id,
        "model_layer_activations_path": model_layer_activations_path,
        "metric": getattr(metric, "name", str(metric)),
    }
    params_path = os.path.join(save_directory, "params.txt")
    with open(params_path, "w") as save_file:
        _write_iterable_to_stream(
            save_file, (f"{param}: {value}" for param, value in params.items())
        )


def format_as_json_string(
    dictionaries_list: Sequence[Mapping[str, Any]],
) -> Sequence[str]:
    """Convert a list of dictionaries to JSON-formatted strings.

    Args:
        dictionaries_list: Sequence of dictionaries to format.

    Returns:
        Sequence of JSON-formatted strings, one per dictionary.
    """
    return [json.dumps(line) for line in dictionaries_list]


def format_best_concepts_history(
    best_concept_history: Sequence[Mapping[str, Any]],
) -> Sequence[str]:
    """Format best concepts history as JSON-formatted strings.

    Args:
        best_concept_history: Sequence of best concept dictionaries.

    Returns:
        Sequence of JSON-formatted strings for best concepts.
    """
    return [json.dumps(line) for line in best_concept_history]


def format_concept_history(
    concept_history: Mapping[str, float],
) -> Sequence[str]:
    """Format concept history as comma-separated concept-score strings.

    Args:
        concept_history: Dictionary mapping concept names to scores.

    Returns:
        Sequence of formatted strings with format 'concept,score'.
    """
    return [f"{concept},{score}" for concept, score in concept_history.items()]

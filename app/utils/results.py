import dataclasses
import os
from typing import Any, Mapping, Sequence, Tuple
import json
import ast


RESULTS_STATE_KEY = "experiment_results"

_IMAGES_DIRECTORY = "images"
_GENERATION_HISTORY_FILE = "generation_history.txt"
_FINAL_CONCEPT_HISTORY_FILE = "final_concept_history.txt"
_PARAMS_FILE = "params.txt"
_BEST_CONCEPTS_FILE = "best_concepts.txt"
_REASONING_FILE = "reasoning.txt"
_REQUIRED_FILES = {
    _PARAMS_FILE,
    _BEST_CONCEPTS_FILE,
    _REASONING_FILE,
    _FINAL_CONCEPT_HISTORY_FILE,
    _GENERATION_HISTORY_FILE,
    _IMAGES_DIRECTORY,
}


@dataclasses.dataclass
class ExperimentResults:
    images: Mapping[str, Sequence[bytes]]
    generation_history: Sequence[Tuple[str, float]]
    final_concept_history: Sequence[Tuple[str, float]]
    pipeline_params: Mapping[str, Any]
    best_concepts: Sequence[Tuple[str, float]]
    reasoning: Sequence[Tuple[str, str]]
    run_params: Mapping[str, Any]


def get_experiment_directories(results_path: str) -> Sequence[str]:
    return [
        filename
        for filename in os.listdir(results_path)
        if os.path.isdir(os.path.join(results_path, filename))
    ]


def _process_history_file(history_filepath: str) -> Sequence[Tuple[str, float]]:
    with open(history_filepath, "r") as history_file:
        history = [tuple(line.split(",")) for line in history_file.readlines()]
    return history


def _listdir_with_absolute(path: str) -> Sequence[str]:
    return [
        (filename, os.path.join(path, filename))
        for filename in os.listdir(path)
    ]


def _process_images_directory(
    images_directory_path: str,
) -> Mapping[str, Sequence[bytes]]:
    images = dict()
    for iteration, images_path in _listdir_with_absolute(images_directory_path):
        if not os.path.isdir(images_path):
            continue
        iteration_images = []
        for _, img_path in _listdir_with_absolute(images_path):
            with open(img_path, "rb") as image:
                iteration_images.append(image)
        images[iteration] = iteration_images
    return images


def _process_params_file(params_filepath: str):
    params = {}
    with open(params_filepath, "r") as params_file:
        for line in params_file:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()

        parsed = ast.literal_eval(params["load_config"])
        params["load_config"] = parsed
    return params


def _process_best_concepts_file(best_concepts_filepath: str):
    best_concepts = []
    with open(best_concepts_filepath, "r") as best_concepts_file:
        for line in best_concepts_file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            best_concepts.append(
                (obj["best_concept"], round(obj["best_score"], 2))
            )
    return best_concepts


def _process_reasoning_file(reasoning_filepath: str):
    reasoning = []
    with open(reasoning_filepath, "r") as reasoning_file:
        for line in reasoning_file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            reasoning.append((obj["concept"], obj["reasoning"]))
    return reasoning


def load_experiment_results(
    results_path: str,
    experiment_directory: str,
) -> ExperimentResults:
    images = generation_history = final_concept_history = params = None
    experiment_directory_path = os.path.join(results_path, experiment_directory)

    if not os.path.exists(experiment_directory_path):
        raise FileNotFoundError(
            f"Directory not found: {experiment_directory_path}"
        )

    existing_files = set(os.listdir(experiment_directory_path))

    missing_files = _REQUIRED_FILES - existing_files
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {experiment_directory_path}: {missing_files}"
        )

    images = {}
    generation_history = []
    final_concept_history = []
    params = {}
    best_concepts = []
    reasoning = []

    if _GENERATION_HISTORY_FILE in existing_files:
        generation_history = _process_history_file(
            os.path.join(experiment_directory_path, _GENERATION_HISTORY_FILE)
        )

    if _FINAL_CONCEPT_HISTORY_FILE in existing_files:
        final_concept_history = _process_history_file(
            os.path.join(experiment_directory_path, _FINAL_CONCEPT_HISTORY_FILE)
        )

    if _IMAGES_DIRECTORY in existing_files:
        images = _process_images_directory(
            os.path.join(experiment_directory_path, _IMAGES_DIRECTORY)
        )

    if _PARAMS_FILE in existing_files:
        params = _process_params_file(
            os.path.join(experiment_directory_path, _PARAMS_FILE)
        )

    if _BEST_CONCEPTS_FILE in existing_files:
        best_concepts = _process_best_concepts_file(
            os.path.join(experiment_directory_path, _BEST_CONCEPTS_FILE)
        )

    if _REASONING_FILE in existing_files:
        reasoning = _process_reasoning_file(
            os.path.join(experiment_directory_path, _REASONING_FILE)
        )

    return ExperimentResults(
        images=images,
        generation_history=generation_history,
        final_concept_history=final_concept_history,
        pipeline_params=params,
        best_concepts=best_concepts,
        reasoning=reasoning,
        run_params=params,
    )

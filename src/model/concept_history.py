"""This module defines logic for initializing and updating the concept history."""

from collections.abc import Iterator
import os
import random
from typing import Mapping, Sequence, Tuple

import torch


def _calculate_average_activation(
    activations: torch.tensor, neuron_id: int
) -> float:
    """Calculate average activation for a selected neuron.

    Args:
        activations: Tensor of activations across samples.
        neuron_id: Index of the neuron to compute average activation for.

    Returns:
        Average activation value for the specified neuron.
    """
    neuron_activations = activations[:, neuron_id]
    return torch.mean(neuron_activations).item()


def _get_activations(
    model_layer_activations_path: str,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Load and iterate over control concept activations from disk.

    Reads activation files from the specified directory and yields concept
    names paired with their corresponding activation tensors.

    Args:
        model_layer_activations_path: Path to directory containing activation files.

    Yields:
        Tuples of concept names and activation tensors.
    """
    for acitvations_filename in os.listdir(model_layer_activations_path):
        filepath = os.path.join(
            model_layer_activations_path, acitvations_filename
        )
        activations = torch.load(filepath)
        concept, _ = os.path.splitext(acitvations_filename)

        yield concept.replace("_", " "), activations


def _create_average_activations(
    model_layer_activations_path: str, neuron_id: int
) -> Mapping[str, float]:
    """Create a mapping of concepts to average neuron activations.

    Calculates the average activation value for a specific neuron across all
    control concepts.

    Args:
        model_layer_activations_path: Path to directory containing activation files.
        neuron_id: Index of the neuron to compute activations for.

    Returns:
        Dictionary mapping concept names to their average activation values.
    """
    average_activations = dict()

    for concept, activations in _get_activations(model_layer_activations_path):
        average_activations[concept] = _calculate_average_activation(
            activations, neuron_id
        )

    return average_activations


def _select_best_concepts(
    average_neuron_activations: Mapping[str, float], n_best_concepts: int
) -> Mapping[str, float]:
    """Select the top concepts by average activation.

    Sorts concepts by their average activation values and returns the
    top n_best_concepts with the highest activations.

    Args:
        average_neuron_activations: Dictionary mapping concepts to activation values.
        n_best_concepts: Number of top concepts to select.

    Returns:
        List of the top concept names sorted by activation (highest first).
    """
    sorted_concepts = sorted(
        average_neuron_activations,
        key=average_neuron_activations.get,
        reverse=True,
    )

    return sorted_concepts[:n_best_concepts]


def get_initial_concepts(
    n_best_concepts: int,
    n_random_concepts: int,
    model_layer_activations_path: str,
    neuron_id: int,
) -> Sequence[str]:
    """Get initial concepts combining best and random selections.

    Returns a sequence of initial concepts consisting of the top n_best_concepts
    that most strongly activate the neuron, plus n_random_concepts randomly
    selected from all available control concepts.

    Args:
        n_best_concepts: Number of top-activating concepts to include.
        n_random_concepts: Number of random concepts to include.
        model_layer_activations_path: Path to directory containing activation files.
        neuron_id: Index of the neuron to select concepts for.

    Returns:
        Sequence of initial concept names combining best and random selections.
    """
    average_neuron_activations = _create_average_activations(
        model_layer_activations_path, neuron_id
    )
    best_concepts = _select_best_concepts(
        average_neuron_activations, n_best_concepts
    )
    random_concepts = random.sample(
        list(average_neuron_activations.keys()), n_random_concepts
    )

    return (*best_concepts, *random_concepts)


def update_concept_history(
    concept_history: Mapping[str, float], new_concept: str, score: float
) -> Mapping[str, float]:
    """Update concept history by replacing the worst or randomly selected concept.

    If the new concept's score is better than the worst existing concept, replaces
    the worst one. Otherwise, randomly removes a concept with probability weighted
    by its distance from the max score.

    Args:
        concept_history: Dictionary mapping concept names to their scores.
        new_concept: Name of the new concept to add.
        score: Score of the new concept.

    Returns:
        Updated concept history dictionary with the new concept added.
    """
    worst_score_concept = min(concept_history, key=concept_history.get)

    if score > concept_history[worst_score_concept]:
        concept_to_remove = worst_score_concept
    else:
        max_score = max(concept_history.values())
        concept_to_remove = random.choices(
            list(concept_history.keys()),
            weights=[max_score - score for score in concept_history.values()],
        ).pop()

    del concept_history[concept_to_remove]
    concept_history[new_concept] = score

    return concept_history

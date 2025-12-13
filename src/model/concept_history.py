"""This module defines logic for initializing and updating the concept history."""

from collections.abc import Iterator
import os
import random
from typing import Mapping, Sequence, Tuple

import torch


def _calculate_average_activation(
    activations: torch.tensor, neuron_id: int
) -> float:
    """Calculates average activation for selected neuron."""
    neuron_activations = activations[:, neuron_id]
    return torch.mean(neuron_activations).item()


def _get_activations(
    model_layer_activations_path: str,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Returns an iterator of concept names and its control activations."""
    for acitvations_filename in os.listdir(model_layer_activations_path):
        filepath = os.path.join(
            model_layer_activations_path, acitvations_filename
        )
        data = torch.load(filepath)
        activations = data["activations"]
        
        concept, _ = os.path.splitext(acitvations_filename)

        yield concept.replace("_", " "), activations


def _create_average_activations(
    model_layer_activations_path: str, neuron_id: int
) -> Mapping[str, float]:
    """
    Creates dictionary with average activation of every control concept for
    selected neuron.
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
    """
    Filters average activations by selecting n_best_concepts concepts with
    the highest average activation.
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
    """
    Gets a list of initial concepts. The list contains n_best_concepts concepts
    that activated the neuron the most and n_random_concepts random concepts.
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
    """Updates the concept history with a new concept."""
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

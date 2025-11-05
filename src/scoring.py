"""This module defines functions for scoring explanations."""

import enum

import sklearn
import torch


def _calculate_auc(
    neuron_control_activations: torch.Tensor,
    neuron_synthetic_activations: torch.Tensor,
) -> float:
    """Calculates AUC between control and synthetic activations."""
    concept_labels = torch.cat(
        (
            torch.zeros([neuron_control_activations.shape[0]]),
            torch.ones([neuron_synthetic_activations.shape[0]]),
        ),
        0,
    )

    activations = torch.cat(
        (neuron_control_activations, neuron_synthetic_activations), 0
    ).to("cpu")

    return sklearn.metrics.roc_auc_score(concept_labels, activations)


def _calculate_average_activation(
    neuron_activations: torch.Tensor,
) -> float:
    """Calculates average neuron activation."""
    return neuron_activations.mean().item()


def _calculate_mad(
    neuron_control_activations: torch.Tensor,
    neuron_synthetic_activations: torch.Tensor,
) -> float:
    """Calculates MAD between synthetic and control activations."""
    average_control_activation = _calculate_average_activation(
        neuron_control_activations
    )
    average_synthetic_activation = _calculate_average_activation(
        neuron_synthetic_activations
    )
    control_std = neuron_control_activations.std().item()

    activations_difference = (
        average_synthetic_activation - average_control_activation
    )
    denominator = control_std if control_std != 0 else 1.0
    return activations_difference / denominator


class Metric(enum.Enum):
    AUC = enum.member(_calculate_auc)
    MAD = enum.member(_calculate_mad)
    AVG_ACTIVATION = enum.member(_calculate_average_activation)


def calculate_metric(
    neuron_control_activations: torch.Tensor,
    neuron_synthetic_activations: torch.Tensor,
    metric: Metric,
) -> float:
    """Calculates selected metric."""
    if (
        neuron_control_activations.ndim != 1
        or neuron_control_activations.ndim != 1
    ):
        raise ValueError("Neuron activations must be one dimensional.")

    if metric == Metric.AVG_ACTIVATION:
        return metric.value(neuron_synthetic_activations)
    else:
        return metric.value(
            neuron_control_activations, neuron_synthetic_activations
        )

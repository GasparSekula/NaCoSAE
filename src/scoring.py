"""Scoring metrics for concept quality evaluation.

Provides various metrics for quantifying how well a concept explains neuron
behavior by comparing activations on synthetic concept images versus
control images.
"""

import enum

import sklearn
import torch


def _calculate_auc(
    neuron_control_activations: torch.Tensor,
    neuron_synthetic_activations: torch.Tensor,
) -> float:
    """Calculate AUC metric for concept scoring.

    Computes the area under the ROC curve treating control activations as
    negative class (0) and synthetic activations as positive class (1).

    Args:
        neuron_control_activations: Activations from control concept images.
        neuron_synthetic_activations: Activations from synthetic concept images.

    Returns:
        AUC score between 0 and 1, higher is better.
    """
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
    """Calculate average neuron activation.

    Args:
        neuron_activations: Activation values for a neuron.

    Returns:
        Mean activation value.
    """
    return neuron_activations.mean().item()


def _calculate_mad(
    neuron_control_activations: torch.Tensor,
    neuron_synthetic_activations: torch.Tensor,
) -> float:
    """Calculate Mean Absolute Difference (MAD) metric.

    Computes the standardized difference between synthetic and control
    activation means, normalized by the control activation standard deviation.

    Args:
        neuron_control_activations: Activations from control concept images.
        neuron_synthetic_activations: Activations from synthetic concept images.

    Returns:
        Standardized activation difference score.
    """
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
    """Enumeration of concept scoring metrics.

    Attributes:
        AUC: Area under ROC curve between control and synthetic activations.
        MAD: Mean absolute difference normalized by control activation standard deviation.
        AVG_ACTIVATION: Average activation magnitude on synthetic images.
    """

    AUC = enum.member(_calculate_auc)
    MAD = enum.member(_calculate_mad)
    AVG_ACTIVATION = enum.member(_calculate_average_activation)


def calculate_metric(
    neuron_control_activations: torch.Tensor,
    neuron_synthetic_activations: torch.Tensor,
    metric: Metric,
) -> float:
    """Calculate the specified metric for a concept.

    Computes the selected scoring metric by comparing control and synthetic
    neuron activations. Different metrics measure different aspects of how
    well a concept explains the neuron.

    Args:
        neuron_control_activations: Neuron activations from control images.
        neuron_synthetic_activations: Neuron activations from synthetic concept images.
        metric: The metric to compute (AUC, MAD, or AVG_ACTIVATION).

    Returns:
        Score value for the selected metric.

    Raises:
        ValueError: If activations are not one-dimensional.
    """
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

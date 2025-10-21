"""This module defines functions for scoring explanations."""

from typing import Mapping

import immutabledict
import sklearn
import torch


def _calculate_auc(
    control_activations: torch.Tensor, synthetic_activations: torch.Tensor
) -> float:
    """
    Calculates AUC between control and synthetic activations.
    It is assumed that these activations are for single neurons only.
    """

    concept_labels = torch.cat(
        (
            torch.zeros([control_activations.shape[0]]),
            torch.ones([synthetic_activations.shape[0]]),
        ),
        0,
    ).to("cpu")

    activations = torch.cat((control_activations, synthetic_activations), 0).to(
        "cpu"
    )

    return sklearn.metrics.roc_auc_score(concept_labels, activations)


def calculate_metrics(
    control_activations: torch.Tensor, synthetic_activations: torch.Tensor
) -> Mapping[str, float]:
    metrics = dict()
    metrics["auc"] = _calculate_auc(control_activations, synthetic_activations)

    return immutabledict.immutabledict(metrics)

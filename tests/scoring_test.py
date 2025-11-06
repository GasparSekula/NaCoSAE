import math
import pytest
import torch

import scoring


def test_calculate_auc_perfect_separation():
    control_activations = torch.tensor([0.1, 0.2, 0.3])
    synthetic_activations = torch.tensor([0.7, 0.6, 0.4])

    auc = scoring._calculate_auc(control_activations, synthetic_activations)
    assert math.isclose(auc, 1.0)


def test_calculate_average_activation():
    neuron_activations = torch.tensor([1.5, 2.5, 2.25, 1.75])

    avg_activation = scoring._calculate_average_activation(neuron_activations)
    assert math.isclose(avg_activation, 2.0)


def test_calculate_mad_zero_std():
    control_activations = torch.tensor([0.5, 0.5, 0.5])
    synthetic_activations = torch.tensor([0.6, 0.8, 0.7])

    mad = scoring._calculate_mad(control_activations, synthetic_activations)
    assert math.isclose(mad, 0.2, rel_tol=1e-6)


def test_calculate_metric_raises_dim_error():
    control_activations = torch.tensor([[1, 2], [3, 4]])
    synthetic_activations = torch.tensor([1, 2, 3, 4])

    with pytest.raises(ValueError):
        scoring.calculate_metric(
            control_activations, synthetic_activations, scoring.Metric.AUC
        )


def test_calculate_metric_avg_activation():
    control_activations = torch.tensor([0.5, 0.5, 0.5])
    synthetic_activations = torch.tensor([0.6, 0.8, 0.7])

    assert math.isclose(
        scoring.calculate_metric(
            control_activations,
            synthetic_activations,
            scoring.Metric.AVG_ACTIVATION,
        ),
        0.7,
        rel_tol=1e-6,
    )

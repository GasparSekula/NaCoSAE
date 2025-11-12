import torch
import pytest_mock

from model import concept_history


def test_calculate_average_activation():
    activations = torch.tensor([[1, 2, 3], [0, 2, -1]], dtype=torch.float)

    assert concept_history._calculate_average_activation(activations, 1) == 2


def test_get_activations(mocker: pytest_mock.MockerFixture):
    test_concept_list = ["concept_1.pt", "concept_2.pt", "concept_3.pt"]
    mocker.patch("os.listdir", return_value=test_concept_list)
    test_tensor = torch.tensor([1, 2, 3])
    mocker.patch("torch.load", return_value=test_tensor)

    result = [
        (concept, activation)
        for concept, activation in concept_history._get_activations("fake/path")
    ]
    expected_result = [
        ("concept 1", test_tensor),
        ("concept 2", test_tensor),
        ("concept 3", test_tensor),
    ]
    assert result == expected_result


def test_create_average_activations(mocker: pytest_mock.MockerFixture):
    mock_get_activations = mocker.patch(
        "model.concept_history._get_activations"
    )
    mock_get_activations.return_value = [
        ("concept_1", torch.tensor([[1, 2], [3, 4]], dtype=torch.float)),
        ("concept_2", torch.tensor([[5, 6], [7, 8]], dtype=torch.float)),
    ]
    expected_result = {"concept_1": 2, "concept_2": 6}

    assert (
        concept_history._create_average_activations("fake_path", 0)
        == expected_result
    )


def test_select_best_concepts():
    average_neuron_activations = {
        "concept_1": 10,
        "concept_2": 3,
        "concept_3": 12,
        "concept_4": 1,
    }
    top_2_concepts = ["concept_3", "concept_1"]

    assert (
        concept_history._select_best_concepts(average_neuron_activations, 2)
        == top_2_concepts
    )


def test_update_concept_history_better_concept():
    history = {
        "concept_1": 0.42,
        "concept_2": 0.83,
        "concept_3": 0.13,
        "concept_4": 0.24,
    }

    score = 0.53
    history = concept_history.update_concept_history(
        history, "concept_5", score
    )

    assert "concept_5" in history
    assert "concept_3" not in history


def test_update_concept_history_worse_concept():
    history = {
        "concept_1": 0.42,
        "concept_2": 0.83,
        "concept_3": 0.13,
        "concept_4": 0.24,
    }

    score = 0.05
    history = concept_history.update_concept_history(
        history, "concept_5", score
    )

    assert "concept_5" in history
    assert len(history) == 4

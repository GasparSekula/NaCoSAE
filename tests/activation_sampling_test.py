from typing import Annotated, Sequence

import pytest
import pytest_mock
import torch

import activation_sampling

ActivationSamplerFixture = Annotated[
    activation_sampling.ActivationSampler, pytest.fixture
]


class MockModel:
    def encode(self, sentences: str | Sequence[str]) -> torch.Tensor:
        if isinstance(sentences, str):
            return torch.tensor([1, 2, 3])
        else:
            return torch.tensor([[1, 2, 3] for _ in range(len(sentences))])

    def similarity(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        return torch.empty((1, embeddings_2.shape[0]))


@pytest.fixture
def activation_sampler(
    mocker: pytest_mock.MockerFixture,
) -> activation_sampling.ActivationSampler:
    mocker.patch(
        "activation_sampling.ActivationSampler._load_model",
        return_value=MockModel(),
    )
    mocker.patch("os.listdir", return_value=["concept_1.pt", "concept_2.pt"])
    mocker.patch(
        "activation_sampling.ActivationSampler._sample_concept_name",
        return_value="concept 1",
    )
    return activation_sampling.ActivationSampler("fake/path")


def test_name_conversions(activation_sampler: ActivationSamplerFixture):
    concept = "concept 1"
    filename = "concept_1.pt"
    assert activation_sampler._filename_to_concept(filename) == concept
    assert activation_sampler._concept_to_filename(concept) == filename


def test_get_similarities_correct_shape(
    activation_sampler: ActivationSamplerFixture,
):
    similarities = activation_sampler._get_similarities("concept")
    assert similarities.shape == torch.Size([1, 2])


def test_sample_control_activations(
    mocker: pytest_mock.MockerFixture,
    activation_sampler: ActivationSamplerFixture,
):
    mock_torch_load = mocker.patch("torch.load")
    activation_sampler.sample_control_activations("concept")

    mock_torch_load.assert_called_once_with("fake/path/concept_1.pt")

from typing import Annotated

import pytest
import pytest_mock
import torch

from model import explained_model

ExplainedModelFixture = Annotated[
    explained_model.ExplainedModel, pytest.fixture
]


class MockExplainedTorchModel(torch.nn.Module):
    def __init__(self) -> None:
        super(MockExplainedTorchModel, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=2, kernel_size=2, stride=2
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.conv.weight = torch.nn.Parameter(
            torch.ones((2, 3, 2, 2), requires_grad=False)
        )
        self.conv.bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.avgpool(x)
        return x


@pytest.fixture
def expl_model(
    mocker: pytest_mock.MockerFixture,
) -> explained_model.ExplainedModel:
    mocker.patch(
        "model.explained_model.ExplainedModel._load",
        return_value=MockExplainedTorchModel().eval(),
    )
    mocker.patch("model.model.Model._send_to_device")
    return explained_model.ExplainedModel(
        "test-model-id", "conv", "test-device", False
    )


def test_get_activations(
    expl_model: ExplainedModelFixture,
    mocker: pytest_mock.MockerFixture,
):
    mocker.patch("torch.cuda.empty_cache")
    mocker.patch(
        "model.explained_model._convert_input",
        lambda input_batch: input_batch,
    )
    n_images = 2
    n_channels = 3
    fake_image_tensors = tuple(
        torch.ones((n_channels, 4, 4)).unsqueeze(0) for _ in range(n_images)
    )
    input_batch = torch.cat(fake_image_tensors, 0)
    activations = expl_model.get_activations(input_batch)
    doubled_activations = expl_model.get_activations(2 * input_batch)

    assert torch.equal(activations, 4 * n_channels * torch.ones((n_images, 2)))
    assert torch.equal(
        doubled_activations, 8 * n_channels * torch.ones((n_images, 2))
    )

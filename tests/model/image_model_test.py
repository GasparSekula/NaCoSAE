from unittest.mock import call, Mock

from typing import Annotated, Any
import pytest
import pytest_mock

from model import image_model


ImageModelFixture = Annotated[image_model.ImageModel, pytest.fixture]

_FAKE_IMAGE = None


@pytest.fixture
def t2i_model(mocker: pytest_mock.MockerFixture) -> image_model.ImageModel:
    mocker.patch("model.image_model.ImageModel._load")
    mocker.patch("model.model.Model._send_to_device")
    return image_model.ImageModel("test-model-id", "test-device", False, 25, 5)


def test_generate_images(
    t2i_model: ImageModelFixture, mocker: pytest_mock.MockerFixture
):
    n_images = 2
    fake_prompt = "fake_prompt"
    mocker.patch(
        "prompts.prompt_utils.concept_image_prompt", return_value=fake_prompt
    )
    mocker.patch("torch.manual_seed", lambda seed: seed)
    fake_model = mocker.patch.object(
        t2i_model,
        "_model",
        Mock(return_value=type("", (object,), {"images": [_FAKE_IMAGE]})()),
    )

    synthetic_images = t2i_model.generate_images(
        n_images, "fake_prompt", "fake_concept"
    )

    fake_model.assert_has_calls(
        [
            call(fake_prompt, generator=0, num_inference_steps=25),
            call(fake_prompt, generator=1, num_inference_steps=25),
        ]
    )
    assert len(synthetic_images) == n_images

import pytest_mock

from prompts import prompt_utils


def test_concept_image_prompt(mocker: pytest_mock.MockerFixture):
    fake_available_descriptions = (("fake angle",), ("fake lightning",))
    fake_prompt_text = "prompt text"
    fake_concept = "concept"
    mocker.patch(
        "prompts.prompt_utils._get_random_visual_descriptions",
        return_value=(
            descriptions[0] for descriptions in fake_available_descriptions
        ),
    )

    expected = "prompt text concept, fake angle, fake lightning"
    actual = prompt_utils.concept_image_prompt(fake_prompt_text, fake_concept)
    assert actual == expected

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


def test_generate_prompt_top_k(mocker: pytest_mock.MockerFixture):
    fake_concept_history = {"concept_1": 1, "concept_2": 2, "concept_3": 3}
    fake_generation_history = ("concept_1,1", "concept_2,2", "concept_3,3")
    fake_prompt_template = "list {concept_list}\nhistory {generation_history}"

    mocker.patch(
        "prompts.prompt_utils._read_prompt_template",
        return_value=fake_prompt_template,
    )

    expected = "list concept_3: 3; concept_2: 2;\nhistory concept_1, concept_2, concept_3"
    actual = prompt_utils.generate_prompt(
        fake_concept_history, fake_generation_history, "prompt_path", 2
    )
    assert actual == expected

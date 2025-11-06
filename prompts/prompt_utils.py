"""Functions for prompt creation."""

from collections.abc import Mapping


def concept_image_prompt(prompt_text: str, concept: str) -> str:
    return " ".join([prompt_text, concept])


def generate_concept_prompt(
    concept_history: Mapping[str, float], prompt_path: str
) -> str:
    # this will require some prompt engineering (xd), temp version
    concept_list = (
        "; ".join(f"{k}: {v}" for k, v in concept_history.items()) + "; "
    )
    with open(prompt_path, "r") as prompt_file:
        text_prompt = prompt_file.read()

    return text_prompt.format(concept_list=concept_list).strip()



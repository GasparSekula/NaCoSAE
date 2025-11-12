"""Functions for prompt creation."""

from collections.abc import Mapping


def concept_image_prompt(prompt_text: str, concept: str) -> str:
    return " ".join([prompt_text, concept])


def generate_concept_prompt(
    concept_history: Mapping[str, float], prompt_path: str
) -> str:
    score_sorted_concepts = (
        f"{k}: {v}"
        for k, v in sorted(
            concept_history.items(), key=lambda item: item[1], reverse=True
        )
    )
    concept_list = "; ".join(score_sorted_concepts) + "; "
    with open(prompt_path, "r") as prompt_file:
        text_prompt = prompt_file.read()

    return text_prompt.format(concept_list=concept_list).strip()

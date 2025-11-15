"""Functions for prompt creation."""

from collections.abc import Mapping, Sequence


def concept_image_prompt(prompt_text: str, concept: str) -> str:
    return " ".join([prompt_text, concept])


def generate_concept_prompt(
    concept_history: Mapping[str, float],
    generation_history: Sequence[str],
    prompt_path: str,
) -> str:
    score_sorted_concepts = (
        f"{k}: {v}"
        for k, v in sorted(
            concept_history.items(), key=lambda item: item[1], reverse=True
        )
    )
    concept_list = "; ".join(score_sorted_concepts) + "; "

    generation_list = []

    for i in range(len(generation_history)):
        bare_concept = generation_history[i].split(",")[0]
        generation_list.append(bare_concept)

    with open(prompt_path, "r") as prompt_file:
        text_prompt = prompt_file.read()

    return text_prompt.format(
        concept_list=concept_list, generation_history=generation_list
    ).strip()

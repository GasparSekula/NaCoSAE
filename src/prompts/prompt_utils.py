"""Functions for prompt creation."""

from collections.abc import Mapping, Sequence
import random
from typing import Iterator

_ANGLES = ("extreme close-up", "wide angle shot", "aerial view", "low angle")
_LIGHTNING = ("cinematic lighting", "natural sunlight", "studio lighting")
_AVAILABLE_DESCRIPTIONS = (_ANGLES, _LIGHTNING)


def _get_random_visual_descriptions() -> Iterator[str]:
    return (
        random.choice(descriptions) for descriptions in _AVAILABLE_DESCRIPTIONS
    )


def concept_image_prompt(prompt_text: str, concept: str) -> str:
    text_to_image_prompt = " ".join([prompt_text, concept])
    random_visual_descriptions = _get_random_visual_descriptions()

    return ", ".join((text_to_image_prompt, *random_visual_descriptions))


def generate_prompt(
    concept_history: Mapping[str, float],
    generation_history: Sequence[str],
    prompt_path: str,
    top_k: int | None = None,
) -> str:
    score_sorted_concepts = (
        f"{k}: {v}"
        for k, v in sorted(
            concept_history.items(), key=lambda item: item[1], reverse=True
        )
    )
    score_sorted_concepts = list(score_sorted_concepts)
    if top_k is not None and top_k > 0:
        score_sorted_concepts = score_sorted_concepts[
            : min(len(score_sorted_concepts), top_k)
        ]

    concept_list = "; ".join(score_sorted_concepts) + "; "
    generation_list = []

    bare_concept = lambda concept_score: concept_score.split(",")[0]
    generation_list = [
        bare_concept(concept_score) for concept_score in generation_history
    ]

    with open(prompt_path, "r") as prompt_file:
        text_prompt = prompt_file.read()

    return text_prompt.format(
        concept_list=concept_list, generation_history=generation_list
    ).strip()

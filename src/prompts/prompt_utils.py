"""Functions for creating and formatting prompts for image and language models.

This module provides utilities to generate prompts for text-to-image models
and language models by combining concepts with visual descriptions and
formatting generation history.
"""

from collections.abc import Mapping, Sequence
import random
from typing import Iterator

_ANGLES = ("extreme close-up", "wide angle shot", "aerial view", "low angle")
_LIGHTNING = ("cinematic lighting", "natural sunlight", "studio lighting")
_AVAILABLE_DESCRIPTIONS = (_ANGLES, _LIGHTNING)


def _get_random_visual_descriptions() -> Iterator[str]:
    """Get random visual descriptions for image generation.

    Returns an iterator that yields one random description from each
    category of visual descriptions (angles and lighting).

    Returns:
        Iterator yielding random visual description strings.
    """
    return (
        random.choice(descriptions) for descriptions in _AVAILABLE_DESCRIPTIONS
    )


def concept_image_prompt(prompt_text: str, concept: str) -> str:
    """Create an image generation prompt by combining text with visual descriptions.

    Combines a base prompt text with a concept and adds random visual descriptions
    (angles and lighting) to guide the image generation model.

    Args:
        prompt_text: Base prompt text for the image.
        concept: Concept name to include in the prompt.

    Returns:
        Complete prompt string for text-to-image generation.
    """
    text_to_image_prompt = " ".join([prompt_text, concept])
    random_visual_descriptions = _get_random_visual_descriptions()

    return ", ".join((text_to_image_prompt, *random_visual_descriptions))


def _read_prompt_template(prompt_path: str) -> str:
    """Read a prompt template from a file.

    Args:
        prompt_path: Path to the prompt template file.

    Returns:
        The contents of the prompt template file as a string.
    """
    with open(prompt_path, "r") as prompt_file:
        prompt_template = prompt_file.read()
    return prompt_template


def generate_prompt(
    concept_history: Mapping[str, float],
    generation_history: Sequence[str],
    prompt_path: str,
    top_k: int | None = None,
) -> str:
    """Generate a prompt for the language model by filling a template.

    Creates a formatted prompt by filling a template with the concept history
    (sorted by score) and generation history. Optionally limits to top-k concepts.

    Args:
        concept_history: Dictionary mapping concept names to their scores.
        generation_history: Sequence of generated concept names.
        prompt_path: Path to the prompt template file.
        top_k: Optional number of top-scoring concepts to include. If None,
               includes all concepts.

    Returns:
        Formatted prompt string with concept and generation history inserted.
    """
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

    concept_list = "; ".join(score_sorted_concepts) + ";"
    generation_list = []

    bare_concept = lambda concept_score: concept_score.split(",")[0]
    generation_list = [
        bare_concept(concept_score) for concept_score in generation_history
    ]

    prompt_template = _read_prompt_template(prompt_path)

    return prompt_template.format(
        concept_list=concept_list, generation_history=", ".join(generation_list)
    ).strip()

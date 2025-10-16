"""Functions for prompt creation."""

def concept_image_prompt(prompt_text: str, concept: str):
    return " ".join([prompt_text, concept])
"""Functions for prompt creation."""

from collections.abc import Mapping


def concept_image_prompt(prompt_text: str, concept: str) -> str:
    return " ".join([prompt_text, concept])


def generate_concept_prompt(concept_history: Mapping[str, float]) -> str:
    # this will require some prompt engineering (xd), temp version
    concept_list = "; ".join(f"{k}: {v}" for k, v in concept_history.items()) + "; "
    template = """
    Propose a new concept based on the concept history. Each concept is scored 
    with a value from 0 to 1, the bigger the better. The history contain 
    scored concepts inside <CONCEPTS>...</CONCEPTS> tags. The format is concept: score, e.g. white: 0.52. 
    Try to infer new concept instead of reusing the ones listed here.

    <CONCEPTS>
    {concept_list}
    </CONCEPTS>
    
    Answer only with the concept.
    """

    return template.format(concept_list=concept_list).strip()


if __name__ == "__main__":
    print(generate_concept_prompt(dict()))

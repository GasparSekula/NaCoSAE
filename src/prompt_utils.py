"""Functions for prompt creation."""

from collections.abc import Mapping


def concept_image_prompt(prompt_text: str, concept: str) -> str:
    return " ".join([prompt_text, concept])


def generate_concept_prompt(concept_history: Mapping[str, float]) -> str:
    # this will require some prompt engineering (xd), temp version
    # hardcoded for testing
    best_list = "stripes: 0.85, animal: 0.67"
    random_list = "fish: 0.45, truck: 0.34"
    template = """
    Propose a new concept based on the concept history. Each concept is scored 
    with a value from 0 to 1, the bigger the better. The history contain best 
    scored concepts inside <BEST>...</BEST> tags and random concepts inside 
    <RANDOM>...</RANDOM> tags. The format is concept: score, e.g. white: 0.52. 
    Try to infer new concept instead of reusing the ones listed here.

    <BEST>
    {best_list}
    </BEST>
    <RANDOM>
    {random_list}
    </RANDOM>
    
    Answer only with the concept.
    """

    return template.format(best_list=best_list, random_list=random_list).strip()


if __name__ == "__main__":
    print(generate_concept_prompt(dict()))

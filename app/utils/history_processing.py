from typing import Sequence, Tuple


def get_scores_from_history(
    generation_history: Sequence[Tuple[str, float]],
) -> Sequence[float]:
    return [concept_score[1] for concept_score in generation_history]

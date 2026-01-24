"""Utility functions for processing experiment history data.

This module provides helper functions for extracting and processing
generation history information from experiment results.
"""

from typing import Sequence, Tuple


def get_scores_from_history(
    generation_history: Sequence[Tuple[str, float]],
) -> Sequence[float]:
    """Extract score values from generation history.

    Extracts the score component from each concept-score tuple in the
    generation history.

    Args:
        generation_history: Sequence of tuples containing concepts and their scores.

    Returns:
        Sequence of float values representing scores from each generation.
    """
    return [concept_score[1] for concept_score in generation_history]

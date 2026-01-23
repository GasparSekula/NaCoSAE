"""Activation sampling utilities for scoring generated concepts.

This module provides functionality to sample control concept activations
based on semantic similarity to newly generated concepts.
"""

import os
import random
from typing import Sequence

import sentence_transformers
import torch

_SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"  # parameterize?


class ActivationSampler:
    """Sample control concept activations based on semantic similarity.

    Uses a sentence transformer to compute embeddings and find similar control
    concepts to a given new concept, then samples from them with inverse
    similarity weighting.
    """

    def __init__(self, model_layer_activations_path: str) -> None:
        """Initialize the activation sampler.

        Args:
            model_layer_activations_path: Path to directory containing control
                concept activation files (.pt files).
        """
        self._model_layer_activations_path = model_layer_activations_path
        self._model = self._load_model()
        self._control_concept_names = self._get_control_concept_names()
        self._control_concept_embeddings = self._model.encode(
            self._control_concept_names
        )

    def _load_model(self) -> sentence_transformers.SentenceTransformer:
        """Load the sentence transformer model.

        Returns:
            Loaded sentence transformer model in evaluation mode.
        """
        return sentence_transformers.SentenceTransformer(
            _SENTENCE_TRANSFORMER
        ).eval()

    def _filename_to_concept(self, filename: str) -> str:
        """Convert activation filename to concept name.

        Args:
            filename: Filename without path (e.g., 'concept_name.pt').

        Returns:
            Concept name with underscores replaced by spaces.
        """
        concept, _ = os.path.splitext(filename)
        return concept.replace("_", " ")

    def _concept_to_filename(self, concept: str) -> str:
        """Convert concept name to activation filename.

        Args:
            concept: Concept name with spaces (e.g., 'concept name').

        Returns:
            Filename with underscores instead of spaces and .pt extension.
        """
        return concept.replace(" ", "_") + ".pt"

    def _get_control_concept_names(self) -> Sequence[str]:
        """Get all control concept names from activation files.

        Returns:
            Sequence of control concept names loaded from disk.
        """
        return tuple(
            self._filename_to_concept(filename)
            for filename in os.listdir(self._model_layer_activations_path)
        )

    def _get_similarities(self, new_concept: str) -> torch.Tensor:
        """Calculate semantic similarities between new and control concepts.

        Computes the cosine similarity between the embedding of a new concept
        and all control concept embeddings.

        Args:
            new_concept: New concept name to compare.

        Returns:
            Tensor of similarity scores.
        """
        new_concept_embedding = self._model.encode(new_concept)

        return self._model.similarity(
            new_concept_embedding, self._control_concept_embeddings
        )

    def _sample_concept_name(self, similarities: torch.Tensor) -> str:
        """Sample a control concept using inverse similarity weighting.

        Uses (1 - similarity) as weights, so less similar concepts are more
        likely to be sampled, encouraging diversity.

        Args:
            similarities: Tensor of similarity scores.

        Returns:
            Sampled control concept name.
        """
        return random.choices(
            self._control_concept_names, weights=(1 - similarities[0, :])
        ).pop()

    def sample_control_activations(self, new_concept: str) -> torch.Tensor:
        """Sample control activations similar to a new concept.

        Finds control concepts semantically similar to the new concept and
        samples one with inverse similarity weighting, then returns its
        activation tensor.

        Args:
            new_concept: New concept to find similar control concepts for.

        Returns:
            Control activation tensor for the sampled control concept.
        """
        similarities = self._get_similarities(new_concept)
        control_concept = self._sample_concept_name(similarities)

        return torch.load(
            os.path.join(
                self._model_layer_activations_path,
                self._concept_to_filename(control_concept),
            )
        )

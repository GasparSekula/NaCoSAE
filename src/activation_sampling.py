"""This module defines logic for sampling control activations for scoring."""

import os
import random
from typing import Sequence

import sentence_transformers
import torch

_SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"  # parameterize?


class ActivationSampler:
    def __init__(self, model_layer_activations_path: str) -> None:
        self._model_layer_activations_path = model_layer_activations_path
        self._model = self._load_model()
        self._control_concept_names = self._get_control_concept_names()
        self._control_concept_embeddings = self._model.encode(
            self._control_concept_names
        )

    def _load_model(self) -> sentence_transformers.SentenceTransformer:
        """Loads sentence transformer model."""
        return sentence_transformers.SentenceTransformer(
            _SENTENCE_TRANSFORMER
        ).eval()

    def _filename_to_concept(self, filename: str) -> str:
        """Converts an activations filename to a concept."""
        concept, _ = os.path.splitext(filename)
        return concept.replace("_", " ")

    def _concept_to_filename(self, concept: str) -> str:
        """Converts a concept to an activations filename."""
        return concept.replace(" ", "_") + ".pt"

    def _get_control_concept_names(self) -> Sequence[str]:
        """Returns control concept names."""
        return tuple(
            self._filename_to_concept(filename)
            for filename in os.listdir(self._model_layer_activations_path)
        )

    def _get_similarities(self, new_concept: str) -> torch.Tensor:
        """Calculates similarities between new concept and control concepts."""
        new_concept_embedding = self._model.encode(new_concept)

        return self._model.similarity(
            new_concept_embedding, self._control_concept_embeddings
        )

    def _sample_concept_name(self, similarities: torch.Tensor) -> str:
        """Samples control concept name with (1 - similarities) as weights."""
        return random.choices(
            self._control_concept_names, weights=(1 - similarities)
        ).pop()

    def sample_control_activations(self, new_concept: str) -> torch.Tensor:
        """Returns a random control activation tensor."""
        similarities = self._get_similarities(new_concept)
        control_concept = self._sample_concept_name(similarities)

        return torch.load(
            os.path.join(
                self._model_layer_activations_path,
                self._concept_to_filename(control_concept),
            )
        )

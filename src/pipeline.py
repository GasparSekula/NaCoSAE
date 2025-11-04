"""This module integrates the models and defines the explanation pipeline."""

import dataclasses
import os
import random
from typing import Any, Literal, Mapping, Sequence, Tuple

from absl import logging
import torch

import image_processing
from model import concept_history
from model import model
from model import explained_model
from model import image_model
from model import language_model
import scoring


@dataclasses.dataclass
class LoadConfig:
    language_model_id: str
    text_to_image_model_id: str
    explained_model_id: str
    language_model_kwargs: Mapping[str, Any]
    text_to_image_model_kwargs: Mapping[str, Any]
    explained_model_kwargs: Mapping[str, Any]


@dataclasses.dataclass
class ImageGenerationConfig:
    n_images: int
    prompt_text: str


@dataclasses.dataclass
class ConceptHistoryConfig:
    n_best_concepts: int
    n_random_concepts: int


def _sample_control_activations(
    concept: str,
    model_layer_activations_path: str,
) -> torch.Tensor:
    """Samples control activations (temp implementation). TODO(piechotam) imp"""
    sampled_activations = random.choice(
        os.listdir(model_layer_activations_path)
    )

    return torch.load(
        os.path.join(model_layer_activations_path, sampled_activations)
    )


class Pipeline:
    def __init__(
        self,
        load_config: LoadConfig,
        image_generation_config: ImageGenerationConfig,
        concept_history_config: ConceptHistoryConfig,
        control_activations_path: str,
        layer: str,
        neuron_id: int,
    ) -> None:
        self._load_config = load_config
        self._image_generation_config = image_generation_config
        self._concept_history_config = concept_history_config
        self._neuron_id = neuron_id
        self._model_layer_activations_path = os.path.join(
            control_activations_path, load_config.explained_model_id, layer
        )

    def _load_models(self) -> None:
        """Loads models to cpu."""
        logging.info("Loading %s." % self._load_config.language_model_id)
        self._lang_model = language_model.LanguageModel(
            model_id=self._load_config.language_model_id,
            device="cpu",
            **self._load_config.language_model_kwargs,
        )
        logging.info("Loading %s." % self._load_config.text_to_image_model_id)
        self._t2i_model = image_model.ImageModel(
            model_id=self._load_config.text_to_image_model_id,
            device="cpu",
            **self._load_config.text_to_image_model_kwargs,
        )
        logging.info("Loading %s." % self._load_config.explained_model_id)
        self._expl_model = explained_model.ExplainedModel(
            model_id=self._load_config.explained_model_id,
            device="cpu",
            **self._load_config.explained_model_kwargs,
        )

    def _get_neuron_activations(
        self, synthetic_input_batch: torch.Tensor, concept: str
    ) -> Sequence[torch.Tensor]:
        """Gets synthetic and control activations of selected neuron."""
        synthetic_activations = self._expl_model.get_activations(
            synthetic_input_batch
        )
        control_activations = _sample_control_activations(
            concept, self._model_layer_activations_path
        )

        return (
            synthetic_activations[:, self._neuron_id],
            control_activations[:, self._neuron_id],
        )

    def _score_concept(self, concept: str) -> Mapping[str, float]:
        synthetic_images = self._t2i_model.generate_images(
            self._image_generation_config.n_images,
            self._image_generation_config.prompt_text,
            concept,
        )
        synthetic_input_batch = image_processing.transform_images(
            self._expl_model.model_id, synthetic_images
        )
        neuron_synthetic_activations, neuron_control_activations = (
            self._get_neuron_activations(
                synthetic_input_batch,
                concept,
            )
        )
        return scoring.calculate_metrics(
            neuron_control_activations, neuron_synthetic_activations
        )

    def _run_iteration(
        self, metric: Literal["auc", "mad"]
    ) -> Tuple[str, float]:
        """Runs single iteration of the explanation pipeline."""
        new_concept = self._lang_model.generate_concept()
        metrics = self._score_concept(
            new_concept,
        )
        self._lang_model.update_concept_history(new_concept, metrics[metric])

        return new_concept, metrics[metric]

    def _initialize_concept_history(self) -> Mapping[str, float]:
        logging.info("Initializing concept history.")
        initial_concepts = concept_history.get_initial_concepts(
            self._concept_history_config.n_best_concepts,
            self._concept_history_config.n_random_concepts,
            self._model_layer_activations_path,
            self._neuron_id,
        )

        return dict(
            (
                concept,
                self._score_concept(
                    concept,
                )["auc"],
            )  # TODO(piechotam) make this less dumb
            for concept in initial_concepts
        )

    def run_pipeline(self, metric: Literal["auc", "mad"], n_iters: int) -> None:
        """Runs the explanation pipeline."""
        self._load_models()
        self._lang_model.set_concept_history(self._initialize_concept_history())

        for iter in range(1, n_iters + 1):
            logging.info("Running iteration %s of %s." % (iter, n_iters))
            new_concept, score = self._run_iteration(metric)
            logging.info(
                "Proposed concept %s with score of %f." % (new_concept, score)
            )

        # somehow retrieve the best scored concept

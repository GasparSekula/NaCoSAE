"""This module integrates the models and defines the explanation pipeline."""

import dataclasses
import datetime
import os
import random
from typing import Any, Mapping, Sequence, Tuple

from absl import logging
from PIL import Image
import torch

import history_managing
import image_processing
from model import concept_history
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


@dataclasses.dataclass
class HistoryManagingConfig:
    save_images: bool
    save_histories: bool
    save_directory: str


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
        history_managing_config: HistoryManagingConfig,
        control_activations_path: str,
        layer: str,
        neuron_id: int,
        metric: scoring.Metric,
    ) -> None:
        self._load_config = load_config
        self._image_generation_config = image_generation_config
        self._concept_history_config = concept_history_config
        self._history_managing_config = history_managing_config
        self._neuron_id = neuron_id
        self._model_layer_activations_path = os.path.join(
            control_activations_path, load_config.explained_model_id, layer
        )
        self._metric = metric
        self._run_id = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
            f"{load_config.explained_model_id}"
            f"{layer}"
            f"{neuron_id}"
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

    def _score_concept(
        self, concept: str, concept_synthetic_images: Sequence[Image.Image]
    ) -> float:
        synthetic_input_batch = image_processing.transform_images(
            self._expl_model.model_id, concept_synthetic_images
        )
        neuron_synthetic_activations, neuron_control_activations = (
            self._get_neuron_activations(
                synthetic_input_batch,
                concept,
            )
        )
        return scoring.calculate_metric(
            neuron_control_activations,
            neuron_synthetic_activations,
            self._metric,
        )

    def _run_iteration(self, iter_number: int) -> Tuple[str, float]:
        """Runs single iteration of the explanation pipeline."""
        new_concept = self._lang_model.generate_concept()
        concept_synthetic_images = self._t2i_model.generate_images(
            self._image_generation_config.n_images,
            self._image_generation_config.prompt_text,
            new_concept,
        )

        if self._history_managing_config.save_images:
            history_managing.save_images_from_iteration(
                concept_synthetic_images,
                os.path.join(
                    self._history_managing_config.save_directory,
                    self._run_id,
                    "images",
                ),
                iter_number,
            )

        score = self._score_concept(new_concept, concept_synthetic_images)
        self._lang_model.update_concept_history(new_concept, score)

        return new_concept, score

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
                    self._t2i_model.generate_images(
                        self._image_generation_config.n_images,
                        self._image_generation_config.prompt_text,
                        concept,
                    ),
                ),
            )
            for concept in initial_concepts
        )

    def _save_histories(self) -> None:
        save_directory = os.path.join(
            self._history_managing_config.save_directory, self._run_id
        )
        history_managing.save_llm_history(
            self._lang_model.generation_history,
            save_directory,
            "generation_history.txt",
        )
        history_managing.save_llm_history(
            self._lang_model.get_formatted_concept_history(),
            save_directory,
            "final_concept_history.txt",
        )
        
        os.makedirs(save_directory, exist_ok=True)
        params = {
            "run_id": self._run_id,
            "load_config": dataclasses.asdict(self._load_config),
            "image_generation_config": dataclasses.asdict(self._image_generation_config),
            "concept_history_config": dataclasses.asdict(self._concept_history_config),
            "history_managing_config": dataclasses.asdict(self._history_managing_config),
            "neuron_id": self._neuron_id,
            "model_layer_activations_path": self._model_layer_activations_path,
            "metric": getattr(self._metric, "name", str(self._metric)),
        }
        params_path = os.path.join(save_directory, "params.txt")
        with open(params_path, "w") as f:
            for key, val in params.items():
                f.write(f"{key}: {val}\n")

    def run_pipeline(self, n_iters: int) -> None:
        """Runs the explanation pipeline."""
        self._load_models()
        self._lang_model.set_concept_history(self._initialize_concept_history())

        for iter_number in range(1, n_iters + 1):
            logging.info("Running iteration %s of %s." % (iter_number, n_iters))
            new_concept, score = self._run_iteration(iter_number)
            logging.info(
                "Proposed concept %s with score of %f." % (new_concept, score)
            )

        if self._history_managing_config.save_histories:
            self._save_histories()

        logging.info(
            "Best concept found is %s with score %f."
            % (self._lang_model.get_best_concept())
        )

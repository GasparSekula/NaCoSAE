"""This module integrates the models and defines the explanation pipeline."""

import dataclasses
import os
import random
from typing import Any, Mapping, Sequence, Literal

from absl import logging
import torch

import image_processing
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


def _load_models(load_config: LoadConfig) -> Sequence[model.Model]:
    """Loads models to cpu."""
    logging.info("Loading %s." % load_config.language_model_id)
    lang_model = language_model.LanguageModel(
        model_id=load_config.language_model_id,
        device="cpu",
        **load_config.language_model_kwargs,
    )
    logging.info("Loading %s." % load_config.text_to_image_model_id)
    t2i_model = image_model.ImageModel(
        model_id=load_config.text_to_image_model_id,
        device="cpu",
        **load_config.text_to_image_model_kwargs,
    )
    logging.info("Loading %s." % load_config.explained_model_id)
    expl_model = explained_model.ExplainedModel(
        model_id=load_config.explained_model_id,
        device="cpu",
        **load_config.explained_model_kwargs,
    )

    return lang_model, t2i_model, expl_model


def _sample_control_activations(
    concept: str,
    control_activations_path: str,
) -> torch.Tensor:
    """Samples control activations (temp implementation)."""
    sampled_activations = random.choice(os.listdir(control_activations_path))

    return torch.load(
        os.path.join(control_activations_path, sampled_activations)
    )


def _get_neuron_activations(
    neuron_id: int,
    expl_model: explained_model.ExplainedModel,
    synthetic_input_batch: torch.Tensor,
    concept: str,
    control_activations_path: str,
) -> Sequence[torch.Tensor]:
    """Gets synthetic and control activations of selected neuron."""
    synthetic_activations = expl_model.get_activations(synthetic_input_batch)
    control_activations = _sample_control_activations(
        concept, control_activations_path
    )

    return (
        synthetic_activations[:, neuron_id],
        control_activations[:, neuron_id],
    )


def _run_iteration(
    lang_model: language_model.LanguageModel,
    t2i_model: image_model.ImageModel,
    expl_model: explained_model.ExplainedModel,
    image_generation_config: ImageGenerationConfig,
    control_activations_path: str,
    neuron_id: int,
    metric: Literal["auc", "mad"],  # TODO(piechotam) make this less dumb
):
    """Runs single iteration of the explanation pipeline."""
    new_concept = lang_model.generate_concept()
    synthetic_images = t2i_model.generate_images(
        image_generation_config.n_images,
        image_generation_config.prompt_text,
        new_concept,
    )
    synthetic_input_batch = image_processing.transform_images(
        expl_model.model_id, synthetic_images
    )
    neuron_synthetic_activations, neuron_control_activations = (
        _get_neuron_activations(
            neuron_id,
            expl_model,
            synthetic_input_batch,
            new_concept,
            control_activations_path,
        )
    )
    metrics = scoring.calculate_metrics(
        neuron_control_activations, neuron_synthetic_activations
    )
    lang_model.update_concept_history(new_concept, metrics[metric])

    return new_concept, metrics[metric]


def run_pipeline(
    load_config: LoadConfig,
    image_generation_config: ImageGenerationConfig,
    control_activations_path: str,
    neuron_id: int,
    metric: Literal["auc", "mad"],
    n_iters: int,
):
    """Runs the explanation pipeline."""
    lang_model, t2i_model, expl_model = _load_models(load_config)

    for iter in range(1, n_iters + 1):
        logging.info("Running iteration %s of %s." % (iter, n_iters))
        new_concept, score = _run_iteration(
            lang_model,
            t2i_model,
            expl_model,
            image_generation_config,
            control_activations_path,
            neuron_id,
            metric,
        )
        logging.info(
            "Proposed concept %s with score of %f." % (new_concept, score)
        )

    # somehow retrieve the best scored concept

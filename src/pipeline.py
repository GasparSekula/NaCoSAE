"""Neuron concept attribution explanation pipeline.

Integrates language models, image generation models, and image classification
models to iteratively generate and refine concept descriptions that explain
individual neuron behavior.
"""

import datetime
import os
from typing import Mapping, Sequence, Tuple

from absl import logging
from PIL import Image
import torch

import activation_sampling
import config
import history_managing
import image_processing
from model import concept_history
from model import explained_model
from model import image_model
from model import language_model
import scoring


class Pipeline:
    """Explanation pipeline for neuron concept attribution.

    Orchestrates an iterative process of generating concepts, creating images,
    and scoring concepts based on their ability to activate target neurons.
    """

    def __init__(
        self,
        load_config: config.LoadConfig,
        image_generation_config: config.ImageGenerationConfig,
        concept_history_config: config.ConceptHistoryConfig,
        history_managing_config: config.HistoryManagingConfig,
        control_activations_path: str,
        layer: str,
        neuron_id: int,
        metric: scoring.Metric,
    ) -> None:
        """Initialize the explanation pipeline.

        Args:
            load_config: Configuration for model loading.
            image_generation_config: Configuration for image generation.
            concept_history_config: Configuration for initial concept selection.
            history_managing_config: Configuration for saving results.
            control_activations_path: Path to control concept activations.
            layer: Layer name of the model to explain.
            neuron_id: Index of the neuron to explain.
            metric: Metric for scoring concepts.
        """
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

        self._reasoning = []
        self._best_concepts = []

        self._save_directory = os.path.join(
            self._history_managing_config.save_directory, self._run_id
        )

        history_managing.save_pipeline_parameters(
            save_directory=self._save_directory,
            run_id=self._run_id,
            load_config=load_config,
            image_generation_config=image_generation_config,
            concept_history_config=concept_history_config,
            history_managing_config=history_managing_config,
            neuron_id=neuron_id,
            metric=metric,
            model_layer_activations_path=self._model_layer_activations_path,
        )

    def _load_models(self) -> None:
        """Load all required models with appropriate device placement.

        Loads the language model, text-to-image model, and image classification
        model. If model swapping is enabled, models are loaded to CPU and moved
        to GPU only during inference. Otherwise, all models are loaded directly
        to GPU.
        """
        initial_device = "cpu" if self._load_config.model_swapping else "cuda"

        logging.info("Loading %s." % self._load_config.language_model_id)
        self._lang_model = language_model.LanguageModel(
            model_id=self._load_config.language_model_id,
            device=initial_device,
            model_swapping=self._load_config.model_swapping,
            **self._load_config.language_model_kwargs,
        )
        logging.info("Loading %s." % self._load_config.text_to_image_model_id)
        self._t2i_model = image_model.ImageModel(
            model_id=self._load_config.text_to_image_model_id,
            device=initial_device,
            model_swapping=self._load_config.model_swapping,
            **self._load_config.text_to_image_model_kwargs,
        )
        logging.info("Loading %s." % self._load_config.explained_model_id)
        self._expl_model = explained_model.ExplainedModel(
            model_id=self._load_config.explained_model_id,
            device=initial_device,
            model_swapping=self._load_config.model_swapping,
            **self._load_config.explained_model_kwargs,
        )

    def _setup(self) -> None:
        """Initialize pipeline components for the explanation process.

        Loads all models, creates the activation sampler, and initializes
        the concept history with best and random control concepts.
        """
        self._load_models()
        self._activation_sampler = activation_sampling.ActivationSampler(
            self._model_layer_activations_path
        )
        self._lang_model.concept_history = self._initialize_concept_history()

    def _get_neuron_activations(
        self, synthetic_input_batch: torch.Tensor, concept: str
    ) -> Sequence[torch.Tensor]:
        """Get target neuron activations from synthetic and control images.

        Computes activations from images generated for a concept and samples
        activations from similar control concepts to compare against.

        Args:
            synthetic_input_batch: Batch of synthetic images generated for the concept.
            concept: Concept name for finding similar control concepts.

        Returns:
            Tuple of (synthetic_activations, control_activations) for the target neuron.
        """
        logging.info("Collecting neuron activations of synthetic images.")
        synthetic_activations = self._expl_model.get_activations(
            synthetic_input_batch
        )

        logging.info("Sampling control activations.")
        control_activations = (
            self._activation_sampler.sample_control_activations(concept)
        )

        return (
            synthetic_activations[:, self._neuron_id],
            control_activations[:, self._neuron_id],
        )

    def _score_concept(
        self, concept: str, concept_synthetic_images: Sequence[Image.Image]
    ) -> float:
        """Score a concept based on neuron activation differences.

        Generates images for a concept, computes neuron activations, and
        compares them to control activations using the specified metric.

        Args:
            concept: Concept name to score.
            concept_synthetic_images: Images generated for the concept.

        Returns:
            Score value computed by the metric.
        """
        logging.info("Scoring proposed concept.")
        synthetic_input_batch = image_processing.transform_images(
            concept_synthetic_images
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

    def _generate_images(self, concept: str) -> Sequence[Image.Image]:
        """Generate images for a given concept.

        Args:
            concept: Concept name to generate images for.

        Returns:
            Sequence of generated PIL Image objects.
        """
        return self._t2i_model.generate_images(
            self._image_generation_config.n_images,
            self._image_generation_config.prompt_text,
            concept,
        )

    def _run_iteration(
        self, iter_number: int, top_k: int | None = None
    ) -> Tuple[str, float]:
        """Run a single iteration of the explanation pipeline.

        Generates a new concept, creates images for it, scores the concept,
        and updates the concept history. Optionally saves images and stores
        reasoning and best concepts.

        Args:
            iter_number: Current iteration number.
            top_k: Optional number of top concepts to use for concept generation.
                   If None, uses all concepts in history.

        Returns:
            Tuple of (new_concept, score) for the generated concept.
        """
        """Runs single iteration of the explanation pipeline."""
        new_concept, reasoning = self._lang_model.generate_concept(top_k)
        concept_synthetic_images = self._generate_images(new_concept)

        if self._history_managing_config.save_images:
            history_managing.save_images_from_iteration(
                concept_synthetic_images,
                os.path.join(
                    self._history_managing_config.save_directory,
                    self._run_id,
                    "images",
                ),
                iter_number,
                new_concept,
            )

        score = self._score_concept(new_concept, concept_synthetic_images)
        self._lang_model.update_concept_history(new_concept, score)
        self._reasoning.append({"concept": new_concept, "reasoning": reasoning})

        best_concept, best_score = self._lang_model.get_best_concept()
        self._best_concepts.append(
            {
                "iteration": iter_number,
                "best_concept": best_concept,
                "best_score": best_score,
            }
        )

        return new_concept, score

    def _initialize_concept_history(self) -> Mapping[str, float]:
        """Initialize concept history with control concepts.

        Scores both the best-activating control concepts and random control
        concepts, returning a dictionary mapping concept names to their scores.

        Returns:
            Dictionary mapping concept names to their initial scores.
        """
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
                self._score_concept(concept, self._generate_images(concept)),
            )
            for concept in initial_concepts
        )

    def _save_histories(self) -> None:
        """Save generation history and final concept history to disk.

        Saves the generation history from the language model and the final
        concept history (all concepts with their final scores) to text files.
        """
        history_managing.save_llm_history(
            self._lang_model.generation_history,
            self._save_directory,
            "generation_history.txt",
        )
        history_managing.save_llm_history(
            history_managing.format_concept_history(
                self._lang_model.concept_history
            ),
            self._save_directory,
            "final_concept_history.txt",
        )

    def run_pipeline(self, n_iters: int) -> None:
        """Run the complete neuron explanation pipeline.

        Initializes components, runs the specified number of concept generation
        and scoring iterations, then a final summary iteration. Saves all
        artifacts including images, reasoning, and best concepts.

        Args:
            n_iters: Number of main iterations to run before the summary iteration.
        """
        """Runs the explanation pipeline."""
        self._setup()

        for iter_number in range(1, n_iters + 1):
            logging.info("RUNNING ITERATION %s OF %s." % (iter_number, n_iters))
            new_concept, score = self._run_iteration(iter_number)
            logging.info(
                "PROPOSED CONCEPT: %s with score of %f." % (new_concept, score)
            )

        logging.info("RUNNING SUMMARY ITERATION.")
        summary_concept, summary_score = self._run_iteration(
            iter_number=n_iters, top_k=3
        )
        logging.info(
            "PROPOSED SUMMARY CONCEPT: %s with score of %f."
            % (summary_concept, summary_score)
        )

        if self._history_managing_config.save_histories:
            self._save_histories()

        logging.info(
            "BEST CONCEPT FOUND: %s with score %f."
            % (self._lang_model.get_best_concept())
        )

        logging.info("SAVING REASONING HISTORY TO FILE.")
        history_managing.save_llm_history(
            history_managing.format_as_json_string(self._reasoning),
            self._save_directory,
            "reasoning.txt",
        )

        logging.info("SAVING BEST CONCEPTS.")
        history_managing.save_llm_history(
            history_managing.format_best_concepts_history(self._best_concepts),
            self._save_directory,
            "best_concepts.txt",
        )

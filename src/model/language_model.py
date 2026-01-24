"""Large language model wrapper for concept generation and reasoning.

This module provides functionality to use a language model to generate and reason
about concepts based on activation patterns and generation history.
"""

from collections.abc import Mapping

import torch
import transformers

from model import concept_history
from model import model
from prompts import prompt_utils


_ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"


class LanguageModel(model.Model):
    """Language model for generating and reasoning about neuron concepts.

    Uses a large language model to generate concept names and explanations
    based on activation history and concept scores.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        model_swapping: bool,
        max_new_tokens: int,
        prompt_path: str,
        summary_prompt_path: str,
    ) -> None:
        """Initialize the language model.

        Args:
            model_id: Identifier of the language model (e.g., 'meta-llama/Llama-2-7b-chat').
            device: Device to load model on ('cuda' or 'cpu').
            model_swapping: Whether to enable model swapping functionality.
            max_new_tokens: Maximum number of tokens to generate per response.
            prompt_path: Path to the prompt template file for generation.
            summary_prompt_path: Path to the prompt template for summary generation.
        """
        super().__init__(model_id, device, model_swapping)
        self._max_new_tokens = max_new_tokens
        self._prompt_path = prompt_path
        self._summary_prompt_path = summary_prompt_path
        self.generation_history = []
        self.concept_history: Mapping[str, float] = {}

    def _load(self, **kwargs) -> torch.nn.Module:
        """Load the language model pipeline.

        Initializes a text-generation pipeline with the specified model and
        configures tokenizer settings for LLaMA models.

        Returns:
            The loaded language model.
        """
        pipeline = transformers.pipeline(
            task="text-generation",
            model=self._model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=self._device,
        )

        if "llama" in self._model_id:
            pipeline.tokenizer.pad_token_id = (
                pipeline.model.config.eos_token_id[0]
            )

        self._pipeline = pipeline  # TODO(piechotam) refactor this assignment
        return pipeline.model

    def update_concept_history(self, new_concept: str, score: float) -> None:
        """Update the concept history with a new concept and score.

        Appends the new concept to the generation history and updates the
        concept history mapping with the new concept.

        Args:
            new_concept: The new concept name to add.
            score: The score associated with the new concept.
        """
        self.generation_history.append(f"{new_concept},{score}")
        self.concept_history = concept_history.update_concept_history(
            self.concept_history, new_concept, score
        )

    @model.gpu_inference_wrapper
    def generate_concept(self, top_k: int | None = None):
        """Generate a new concept name with reasoning using the language model.

        Uses the concept history and generation history to generate a new concept
        name. Optionally uses a summary prompt with top-k concepts for faster
        iteration. Parses the model response to extract both the answer (concept)
        and reasoning.

        Args:
            top_k: Optional number of top concepts to use in summary mode.
                   If None, uses the full prompt template.

        Returns:
            Tuple of (concept_name, reasoning_text) from the model.
        """
        self._pipeline.device = torch.device("cuda")

        if top_k is not None:
            generation_prompt = prompt_utils.generate_prompt(
                self.concept_history,
                self.generation_history,
                self._summary_prompt_path,
                top_k,
            )
        else:
            generation_prompt = prompt_utils.generate_prompt(
                self.concept_history, self.generation_history, self._prompt_path
            )

        # only llama for now
        message = self._pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": generation_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        terminators = [
            self._pipeline.tokenizer.eos_token_id,
            self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        output = self._pipeline(
            message,
            max_new_tokens=self._max_new_tokens,
            eos_token_id=terminators,
            temperature=0.5,
            top_p=0.9,
        )
        response = output[0]["generated_text"].split(_ASSISTANT_TAG)[1].strip()

        reasoning = (
            response.split("<thinking>")[1].split("</thinking>")[0].strip()
        )
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()

        return answer, reasoning

    def get_best_concept(self) -> str:
        """Get the best concept and its score from the history.

        Returns the concept with the highest score from the concept history.

        Returns:
            Tuple of (best_concept_name, score) for the highest-scoring concept.
        """
        best_concept = max(self.concept_history, key=self.concept_history.get)
        score = self.concept_history[best_concept]

        return best_concept, score

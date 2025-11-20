from collections.abc import Mapping
from typing import Sequence

import torch
import transformers

from model import concept_history
from model import model
from prompts import prompt_utils


_ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"


class LanguageModel(model.Model):
    def __init__(
        self,
        model_id: str,
        device: str,
        max_new_tokens: int,
        prompt_path: str,
        summary_prompt_path: str,
    ) -> None:
        super().__init__(model_id, device)
        self._max_new_tokens = max_new_tokens
        self._prompt_path = prompt_path
        self._summary_prompt_path = summary_prompt_path
        self.generation_history = []
        self.concept_history: Mapping[str, float] = {}

    def _load(self) -> None:
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

        self._pipeline = pipeline
        self._model = pipeline.model

    def update_concept_history(self, new_concept: str, score: float) -> None:
        """Updates concept history with most recent concept."""
        self.generation_history.append(f"{new_concept},{score}")
        self.concept_history = concept_history.update_concept_history(
            self.concept_history, new_concept, score
        )

    @model.gpu_inference_wrapper
    def generate_concept(self, top_k: int | None = None):
        """Generates new concept based on concept history."""
        self._pipeline.device = torch.device("cuda")  # TODO(piechotam) inv

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
        return response

    def get_best_concept(self) -> str:
        """Returns the best proposed concept."""
        best_concept = max(self.concept_history, key=self.concept_history.get)
        score = self.concept_history[best_concept]

        return best_concept, score

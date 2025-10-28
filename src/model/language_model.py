import torch
import transformers

from model import model
import prompt_utils

_ASSISTANT_TAG = "<|assistant|>"


class LanguageModel(model.Model):
    def __init__(
        self,
        model_id: str,
        device: str,
        n_best_concepts: int,
        n_random_concepts: int,
        max_new_tokens: int,
    ) -> None:
        super().__init__(model_id, device)
        self._load()
        self._initialize_concept_history(n_best_concepts, n_random_concepts)
        self._max_new_tokens = max_new_tokens

    def _load(self) -> None:
        pipeline = transformers.pipeline(
            task="text-generation",
            model=self._model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=self._device,
        )

        if "llama" in self._model_id:
            pipeline.tokenizer.pad_token_id = (
                pipeline.model.config.eos_token_id[0]
            )

        self._pipeline = pipeline

    def _initialize_concept_history(
        self, n_best_concepts: int, n_random_concepts: int
    ) -> None:
        """Initializes concept history with random concepts from control set."""
        # sample random imagenet classes, score them and create a dict
        # best: {concept: score}
        # random: {concept: score}
        self._concept_history = dict()

    def _update_concept_history(
        self, new_concept: str, auc_score: float
    ) -> None:
        """Updates concept history with most recent concept."""
        pass

    def generate_concept(self):
        """Generates new concept based on concept history."""
        concept_generation_prompt = prompt_utils.generate_concept_prompt(
            self._concept_history
        )
        # only llama for now
        message = self._pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": concept_generation_prompt}],
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

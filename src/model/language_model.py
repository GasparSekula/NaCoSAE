import torch
import transformers

from model import model
import prompt_utils
from collections.abc import Mapping
import random


_ASSISTANT_TAG = "<|assistant|>"
_RANDOM_SEED = 42
_MAX_CONCEPT_DICT_SIZE = 100

def select_concepts(concepts_dict: Mapping[str, float], 
                    new_concept: str, 
                    new_concept_score: float,
                    p: float = 0.9) -> Mapping[str, float]:
    
    if not concepts_dict:
        return {new_concept: new_concept_score}
    
    if not (new_concept and new_concept_score):
        return concepts_dict
    
    worst_concept = min(concepts_dict, key=concepts_dict.get)
    
    rng = random.Random(_RANDOM_SEED)
    
    new_dict = dict(concepts_dict)

    if new_concept in new_dict:
        new_dict[new_concept] = max(new_dict[new_concept], new_concept_score)
        return new_dict

    if len(new_dict) < _MAX_CONCEPT_DICT_SIZE:
        new_dict[new_concept] = new_concept_score
        return new_dict

    if new_concept_score > new_dict[worst_concept]:
        del new_dict[worst_concept]
        new_dict[new_concept] = new_concept_score
        return new_dict

    if rng.random() > p:
        replace_key = rng.choice(list(new_dict.keys()))
        del new_dict[replace_key]
        new_dict[new_concept] = new_concept_score

    return new_dict
    
        

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
        
        # hardcoded for testing purposes 
        # TODO: implement initialization
        
        best = {"cup" : 0.9, "cafe": 0.86, "tea" : 0.78, "juice": 0.71, "hot": 0.58}
        random = {"road": 0.32, "beer": 0.71, "horse": 0.22, "cake": 0.49, "toothpaste": 0.13}
        concept_dict = {**best, **random}
        
        self._concept_history = dict(concept_dict)


    def _update_concept_history(
        self, new_concept: str, auc_score: float
    ) -> None:
        """Updates concept history with most recent concept."""
        new_concept_dict = select_concepts(
            self._concept_history,
            new_concept,
            auc_score
        )
        
        self._concept_history = new_concept_dict
        

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

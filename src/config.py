"""This module defines configurations for the pipeline."""

import dataclasses
from typing import Any, Mapping


@dataclasses.dataclass
class LoadConfig:
    language_model_id: str
    text_to_image_model_id: str
    explained_model_id: str
    language_model_kwargs: Mapping[str, Any]
    text_to_image_model_kwargs: Mapping[str, Any]
    explained_model_kwargs: Mapping[str, Any]
    model_swapping: bool


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

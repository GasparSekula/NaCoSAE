"""Configuration dataclasses for the pipeline.

Defines configuration dataclasses for model loading, image generation,
concept history, and result management.
"""

import dataclasses
from typing import Any, Mapping


@dataclasses.dataclass
class LoadConfig:
    """Configuration for loading machine learning models.

    Attributes:
        language_model_id: Identifier for the language model.
        text_to_image_model_id: Identifier for the text-to-image model.
        explained_model_id: Identifier for the model to explain.
        language_model_kwargs: Additional arguments for language model initialization.
        text_to_image_model_kwargs: Additional arguments for text-to-image model initialization.
        explained_model_kwargs: Additional arguments for explained model initialization.
        model_swapping: Whether to enable GPU/CPU model swapping for memory efficiency.
    """

    language_model_id: str
    text_to_image_model_id: str
    explained_model_id: str
    language_model_kwargs: Mapping[str, Any]
    text_to_image_model_kwargs: Mapping[str, Any]
    explained_model_kwargs: Mapping[str, Any]
    model_swapping: bool


@dataclasses.dataclass
class ImageGenerationConfig:
    """Configuration for image generation.

    Attributes:
        n_images: Number of images to generate per concept.
        prompt_text: Base prompt text for image generation.
    """

    n_images: int
    prompt_text: str


@dataclasses.dataclass
class ConceptHistoryConfig:
    """Configuration for concept history initialization.

    Attributes:
        n_best_concepts: Number of top-activating control concepts to include initially.
        n_random_concepts: Number of random control concepts to include initially.
    """

    n_best_concepts: int
    n_random_concepts: int


@dataclasses.dataclass
class HistoryManagingConfig:
    """Configuration for saving experiment results.

    Attributes:
        save_images: Whether to save generated images to disk.
        save_histories: Whether to save generation and concept history to disk.
        save_directory: Directory path where results will be saved.
    """

    save_images: bool
    save_histories: bool
    save_directory: str

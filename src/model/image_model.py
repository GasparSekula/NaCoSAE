"""Text-to-image model for generating synthetic images from prompts.

This module provides functionality to load and use diffusion-based text-to-image
models for generating synthetic images based on text prompts and concepts.
"""

from collections.abc import Sequence

from absl import logging
import diffusers
import immutabledict
from PIL import Image
import torch

from model import model
from prompts import prompt_utils

_TEXT_TO_IMAGE_MODELS = immutabledict.immutabledict(
    {
        "stabilityai/sd-turbo": diffusers.AutoPipelineForText2Image,
        "stabilityai/sdxl-turbo": diffusers.AutoPipelineForText2Image,
        "stabilityai/stable-diffusion-xl-base-1.0": diffusers.DiffusionPipeline,
    }
)
_TORCH_DTYPE = torch.float16
_PIPELINE_VARIANT = "fp16"


class ImageModel(model.Model):
    """Text-to-image generation model wrapper.

    Wraps diffusion-based text-to-image models to generate synthetic images
    from text prompts using specified configurations.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        model_swapping: str,
        num_inference_steps: int,
        guidance_scale: int,
    ) -> None:
        """Initialize the image generation model.

        Args:
            model_id: Identifier of the text-to-image model (e.g., 'stabilityai/sd-turbo').
            device: Device to load model on ('cuda' or 'cpu').
            model_swapping: Whether to enable model swapping functionality.
            num_inference_steps: Number of denoising steps for image generation.
            guidance_scale: Classifier-free guidance scale for conditioning strength.
        """
        super().__init__(
            model_id, device, model_swapping, guidance_scale=guidance_scale
        )
        self._num_inference_steps = num_inference_steps

    def _load(self, guidance_scale: int) -> diffusers.DiffusionPipeline:
        """Load the text-to-image diffusion pipeline.

        Loads the appropriate diffusion pipeline for the specified model and
        configures it with the given guidance scale.

        Args:
            guidance_scale: Classifier-free guidance scale for conditioning.

        Returns:
            Loaded diffusion pipeline on the specified device.
        """
        model = (
            _TEXT_TO_IMAGE_MODELS[self._model_id]
            .from_pretrained(
                self._model_id,
                torch_dtype=_TORCH_DTYPE,
                variant=_PIPELINE_VARIANT,
                guidance_scale=guidance_scale,
            )
            .to(self._device)
        )

        return model

    @model.gpu_inference_wrapper
    def generate_images(
        self, n_images: int, prompt_text: str, concept: str
    ) -> Sequence[Image.Image]:
        """Generate synthetic images for a given concept.

        Generates multiple images by passing text prompts combined with a concept
        through the text-to-image model. Each image uses a different random seed
        for diversity.

        Args:
            n_images: Number of images to generate.
            prompt_text: Base prompt text to use for generation.
            concept: Concept name to include in the image prompt.

        Returns:
            Sequence of PIL Image objects generated from the prompts.
        """
        text_to_image_prompt = prompt_utils.concept_image_prompt(
            prompt_text, concept
        )

        logging.info(f"Generating %d images of %s." % (n_images, concept))
        synthetic_images = []
        for i in range(n_images):
            generator = torch.manual_seed(i)
            synthetic_images.append(
                self._model(
                    text_to_image_prompt,
                    generator=generator,
                    num_inference_steps=self._num_inference_steps,
                ).images[0]
            )

        return tuple(synthetic_images)

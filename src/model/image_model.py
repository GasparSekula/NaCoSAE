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
    def __init__(
        self,
        model_id: str,
        device: str,
        num_inference_steps: int,
        guidance_scale: int,
    ) -> None:
        super().__init__(model_id, device)
        self._guidance_scale = guidance_scale
        self._num_inference_steps = num_inference_steps

    def _load(self) -> diffusers.DiffusionPipeline:
        model = (
            _TEXT_TO_IMAGE_MODELS[self._model_id]
            .from_pretrained(
                self._model_id,
                torch_dtype=_TORCH_DTYPE,
                variant=_PIPELINE_VARIANT,
                guidance_scale=self._guidance_scale,
            )
            .to(self._device)
        )

        return model

    @model.gpu_inference_wrapper
    def generate_images(
        self, n_images: int, prompt_text: str, concept: str
    ) -> Sequence[Image.Image]:
        """Generates images."""
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

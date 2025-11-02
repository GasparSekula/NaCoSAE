from collections.abc import Sequence

import diffusers
import immutabledict
import PIL
import torch

from model import model
import prompt_utils

_TEXT_TO_IMAGE_MODELS = immutabledict.immutabledict(
    {"stabilityai/sd-turbo": diffusers.AutoPipelineForText2Image}
)
_TORCH_DTYPE = torch.float16
_PIPELINE_VARIANT = "fp16"


class ImageModel(model.Model):
    def __init__(
        self, model_id: str, device: str, num_inference_steps: int
    ) -> None:
        super().__init__(model_id, device)
        self._generator = torch.Generator(device).manual_seed(0)
        self._num_inference_steps = num_inference_steps

    def _load(self) -> None:
        model = (
            _TEXT_TO_IMAGE_MODELS[self._model_id]
            .from_pretrained(
                self._model_id,
                torch_dtype=_TORCH_DTYPE,
                variant=_PIPELINE_VARIANT,
            )
            .to(self._device)
        )

        self._model = model

    @model.gpu_inference_wrapper
    def generate_images(
        self, n_images: int, prompt_text: str, concept: str
    ) -> Sequence[PIL.Image.Image]:
        """Generates images."""
        text_to_image_prompt = prompt_utils.concept_image_prompt(
            prompt_text, concept
        )

        synthetic_images = []
        for _ in range(n_images):
            synthetic_images.append(
                self._model(
                    text_to_image_prompt,
                    generator=self._generator,
                    num_inference_steps=self._num_inference_steps,
                ).images[0]
            )

        return tuple(synthetic_images)

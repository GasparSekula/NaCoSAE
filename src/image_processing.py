"""Module for image processing tasks."""

from collections.abc import Sequence

import immutabledict
from PIL import Image
import torch
import torchvision

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class ConditionalNormalize(object):
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, tensor):
        """
        Skip normalization if called on greyscale image and enforce 3 channels.
        """
        if tensor.shape[0] == 3:
            return self.normalize(tensor)
        else:
            return torch.concat((tensor, tensor, tensor), 0)


_TRANSFORMS = immutabledict.immutabledict(
    {
        "resnet18": torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop((224, 224)),
                torchvision.transforms.ToTensor(),
                ConditionalNormalize(_MEAN, _STD),
            ]
        )
    }
)


def transform_images(
    model_id: str,
    images: Sequence[Image.Image],
) -> torch.Tensor:
    """Transforms images and return input batch for explained model."""
    image_tensors = []
    transform = _TRANSFORMS[model_id]

    for image in images:
        image_tensors.append(transform(image).unsqueeze(0))

    batch = torch.concat(tuple(image_tensors), 0)
    return batch


def save_images_from_iteration(
    directory_path: str,
    images: Sequence[Image.Image],
    run_id: str,
    iteration: int,
) -> None:
    """Saves images from an iteration to a specified directory."""
    for image_number, image in enumerate(images):
        filename = f"{run_id}_{iteration}_{image_number}"
        save_path = f"{directory_path}/{filename}"
        image.save(save_path, format="jpeg")

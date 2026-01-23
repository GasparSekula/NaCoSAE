"""Image preprocessing and transformation utilities.

Provides functions and classes for resizing, normalizing, and transforming
images into tensors suitable for neural network inference.
"""

from collections.abc import Sequence

from PIL import Image
import torch
import torchvision

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class ConditionalNormalize(object):
    """Normalize image tensor, handling greyscale by replicating channels.

    Converts single-channel (greyscale) images to 3-channel by replicating
    the channel, then applies standard ImageNet normalization.
    """

    def __init__(self, mean, std):
        """Initialize the normalizer.

        Args:
            mean: Mean values for each channel for normalization.
            std: Standard deviation values for each channel for normalization.
        """
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply channel replication and normalization to tensor.

        If the input tensor has a single channel (greyscale), it is converted
        to 3 channels by replicating the channel. Then standard normalization
        is applied using the configured mean and standard deviation.

        Args:
            tensor: Input tensor to normalize.

        Returns:
            Normalized tensor with 3 channels.
        """
        if tensor.shape[0] == 1:
            tensor = torch.concat((tensor, tensor, tensor), 0)

        return self.normalize(tensor)


_TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        ConditionalNormalize(_MEAN, _STD),
    ]
)


def transform_images(images: Sequence[Image.Image]) -> torch.Tensor:
    """Transform images into a batch tensor for model inference.

    Applies standard preprocessing: resizing to 224x224, center cropping,
    conversion to tensor, and normalization. Returns a batched tensor.

    Args:
        images: Sequence of PIL Image objects to transform.

    Returns:
        Batched tensor of shape (N, 3, 224, 224) ready for inference.

    Raises:
        ValueError: If images sequence is None or empty.
    """
    if images is None or len(images) == 0:
        raise ValueError(f"Expected non-empty sequence of images.")

    image_tensors = []

    for image in images:
        image_tensors.append(_TRANSFORM(image).unsqueeze(0))

    batch = torch.concat(tuple(image_tensors), 0)
    return batch


def save_images_from_iteration(
    directory_path: str,
    images: Sequence[Image.Image],
    run_id: str,
    iteration: int,
) -> None:
    """Save images from an iteration to disk in JPEG format.

    Args:
        directory_path: Directory path to save images to.
        images: Sequence of PIL Image objects to save.
        run_id: Unique run identifier to include in filenames.
        iteration: Iteration number to include in filenames.
    """
    for image_number, image in enumerate(images):
        filename = f"{run_id}_{iteration}_{image_number}"
        save_path = f"{directory_path}/{filename}"
        image.save(save_path, format="jpeg")

from PIL import Image
import pytest
import torch

import image_processing


def test_conditional_normalize():
    grayscale_image_tensor = torch.ones((1, 3, 3))
    rgb_image_tensor = torch.ones((3, 3, 3))
    mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    conditional_normalize = image_processing.ConditionalNormalize(mean, std)

    normalized_rgb_tensor = conditional_normalize(rgb_image_tensor)
    normalized_greyscale_tensor = conditional_normalize(grayscale_image_tensor)
    expected_output = torch.ones(3, 3, 3) * 2

    assert torch.equal(normalized_rgb_tensor, expected_output)
    assert torch.equal(normalized_greyscale_tensor, expected_output)


def test_transform_images():
    num_images = num_channels = 3
    height = width = 10
    resized_height = resized_width = 224
    images = tuple(Image.new("RGB", (height, width)) for _ in range(num_images))
    batch = image_processing.transform_images(images)

    assert batch.shape == (
        num_images,
        num_channels,
        resized_height,
        resized_width,
    )


def test_transform_images_missing_images():
    with pytest.raises(ValueError):
        image_processing.transform_images([])

    with pytest.raises(ValueError):
        image_processing.transform_images(None)

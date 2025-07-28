import torch
import torchvision
from PIL import Image
from typing import Union

class Scorer:
    def __init__(
        self,
        sparse_autoencoder: torch.nn.Module,
        encoding_model: torch.nn.Module,
        image_preprocess: torchvision.transforms.Compose,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """
        Initializes the Scorer.
        """
        self._device = device
        self._sparse_autoencoder = sparse_autoencoder.to(device).eval()
        self._encoding_model = encoding_model.to(device).eval()
        self._image_preprocess = image_preprocess

    @torch.no_grad()
    def __call__(self, image: Image.Image, neuron_index: int) -> list[float]:
        """
        Scores image based on a neuron's activation.

        [!] Temporary solution as there is no sae and encoding model implementation yet
        """
        image_tensor = self._image_preprocess(image).to(self._device)
        image_encoding = self._encoding_model(image_tensor)
        sae_activations = self._sparse_autoencoder(image_encoding).squeeze()
        score = sae_activations[neuron_index]

        return score
"""Explained model for capturing intermediate layer activations.

This module provides functionality to load pretrained models and capture
activations from specific layers for neuron interpretability analysis.
"""

import immutabledict
import torch
import torchvision

from model import model

_WEIGHTS = immutabledict.immutabledict(
    {
        "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": None,  # ResNet50 will use Places365 weights which are not available in torchvision
        "vit_b_16": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,
    }
)

_OPTIONAL_KWARGS = immutabledict.immutabledict(
    {"resnet50": {"num_classes": 365}}
)

_PLACES_365_MODELS = ("resnet50",)
_PLACES_365_WEIGHTS_URL = (
    "http://places2.csail.mit.edu/models_places365/{model_id}_places365.pth.tar"
)


def _load_places_365_state_dict(model_id: str):
    """Load Places365 pretrained weights from remote URL.

    Downloads and loads the state dict for Places365-pretrained models,
    removing the 'module.' prefix added by DataParallel training.

    Args:
        model_id: Identifier of the model (e.g., 'resnet50').

    Returns:
        State dictionary with cleaned keys suitable for loading into a model.
    """
    checkpoint = torch.hub.load_state_dict_from_url(
        _PLACES_365_WEIGHTS_URL.format(model_id=model_id)
    )
    state_dict = checkpoint["state_dict"]
    new_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }

    return new_state_dict


def _convert_input(input_batch: torch.Tensor) -> torch.Tensor:
    """Convert input tensor to float and move to CUDA device.

    Args:
        input_batch: Input tensor to convert.

    Returns:
        Converted tensor as float on CUDA device.
    """
    return input_batch.float().to("cuda")


class ExplainedModel(model.Model):
    """Model wrapper that captures and returns intermediate layer activations.

    Extends the base Model class by registering forward hooks on a specified
    layer to capture neuron activations for interpretation and analysis.
    """

    def __init__(
        self, model_id: str, layer: str, device: str, model_swapping: bool
    ) -> None:
        """Initialize the explained model.

        Args:
            model_id: Identifier of the model architecture (e.g., 'resnet18', 'vit_b_16').
            layer: Name of the layer to capture activations from.
            device: Device to load model on ('cuda' or 'cpu').
            model_swapping: Whether to enable model swapping functionality.

        Raises:
            ValueError: If the specified layer is not found in the model.
        """
        super().__init__(model_id, device, model_swapping)
        self._register_forward_hook(layer)

    def _load(self, **kwargs):
        """Load and initialize the pretrained model.

        Loads the appropriate pretrained weights based on model_id. Uses ImageNet
        weights for most models, or Places365 weights for models like ResNet50.

        Returns:
            Loaded model on the specified device in evaluation mode.
        """
        weights = _WEIGHTS[self._model_id]
        model = torchvision.models.get_model(
            self._model_id,
            weights=weights,
            **_OPTIONAL_KWARGS.get(self._model_id, dict()),
        )

        if self._model_id in _PLACES_365_MODELS:
            state_dict = _load_places_365_state_dict(self._model_id)
            model.load_state_dict(state_dict)

        return model.to(self._device).eval()

    def _hook(self, module, input, output) -> None:
        """Store layer output activations during forward pass.

        This is called by the forward hook to capture and store the output
        of the registered layer.

        Args:
            module: The module that triggered the hook.
            input: The input to the module.
            output: The output of the module.
        """
        self._activations: torch.Tensor = output

    def _register_forward_hook(self, layer: str) -> None:
        """Register a forward hook on the specified layer.

        Registers a hook that captures activations from the specified layer
        during forward passes through the model.

        Args:
            layer: Name of the layer to register the hook on.

        Raises:
            ValueError: If the layer name is not found in the model.
        """
        named_children_dict = {
            name: child for (name, child) in self._model.named_children()
        }
        if layer not in named_children_dict:
            raise ValueError(f"Layer {layer} not found in the model.")

        named_children_dict[layer].register_forward_hook(
            lambda *args, **kwargs: ExplainedModel._hook(self, *args, **kwargs)
        )

    @model.gpu_inference_wrapper
    def get_activations(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through model and return layer activations.

        Passes input through the model and captures activations from the
        registered layer. Handles different activation shapes for CNN and
        Vision Transformer architectures, performing appropriate pooling.

        Args:
            input_batch: Input tensor of shape (N, C, H, W).

        Returns:
            Neuron activations of shape (N, num_neurons).

        Raises:
            ValueError: If input_batch is not 4-dimensional or if activations
                have unexpected dimensionality.
        """
        if input_batch.ndim != 4:
            raise ValueError(
                f"input_batch must be of shape (N, C, H, W)."
                "Provided input_batch has {input_batch.ndim} dimensions."
            )

        with torch.no_grad():
            torch.cuda.empty_cache()
            _ = self._model(_convert_input(input_batch))

        if self._activations.ndim == 4:  # CNN
            self._activations = self._activations.mean(dim=[2, 3])
        elif self._activations.ndim == 3:  # ViT
            self._activations = self._activations[:, 0]
        elif self._activations.ndim != 2:
            raise ValueError(
                f"Explained Model generated activations of unexpected ndim: "
                "{self._activations.ndim}. Expected ndim to be 2, 3 or 4."
            )

        return self._activations.data

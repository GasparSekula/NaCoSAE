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
    checkpoint = torch.hub.load_state_dict_from_url(
        _PLACES_365_WEIGHTS_URL.format(model_id=model_id)
    )
    state_dict = checkpoint["state_dict"]
    new_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }

    return new_state_dict


def _convert_input(input_batch: torch.Tensor) -> torch.Tensor:
    return input_batch.float().to("cuda")


class ExplainedModel(model.Model):
    def __init__(
        self, model_id: str, layer: str, device: str, model_swapping: bool
    ) -> None:
        super().__init__(model_id, device, model_swapping)
        self._register_forward_hook(layer)

    def _load(self, **kwargs):
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
        self._activations: torch.Tensor = output

    def _register_forward_hook(self, layer: str) -> None:
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
        """Passes the input batch through the model and collects activations."""
        with torch.no_grad():
            torch.cuda.empty_cache()
            _ = self._model(_convert_input(input_batch))

        if self._activations.ndim == 4:  # CNN
            self._activations = self._activations.mean(dim=[2, 3])
        elif self._activations.ndim == 3:  # ViT
            self._activations = self._activations[:, 0]

        return self._activations.data

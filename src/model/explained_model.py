import immutabledict
import torch
import torchvision

from model import model

_WEIGHTS = immutabledict.immutabledict(
    {
        "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
    }
)
_CONSTRUCTORS = immutabledict.immutabledict(
    {
        "resnet18": torchvision.models.resnet18,
    }
)


def _convert_input(input_batch: torch.Tensor) -> torch.Tensor:
    return input_batch.float().to("cuda")


class ExplainedModel(model.Model):
    def __init__(self, model_id: str, layer: str, device: str) -> None:
        super().__init__(model_id, device)
        self._register_forward_hook(layer)

    def _load(self):
        weights = _WEIGHTS[self._model_id]
        model = _CONSTRUCTORS[self._model_id](weights=weights)

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

        if self._activations.ndim == 4:
            self._activations = self._activations.mean(dim=[2, 3])

        return self._activations.data

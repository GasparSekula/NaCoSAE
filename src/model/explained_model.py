import immutabledict
import torch
import torchvision

from src.model import model

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


class ExplainedModel(model.Model):
    def __init__(self, model_id: str, device: str) -> None:
        super().__init__(model_id, device)
        self._load()
    
    def _load(self):
        weights = _WEIGHTS[self._model_id]
        self._model = (
            _CONSTRUCTORS[self._model_id](weights=weights)
            .to(self._device)
            .eval()
        )

    def get_activations(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Passes the input batch through the model and collects activations."""
        activations = dict()

        def hook(model, input, output):
            activations["output"] = output

        # TODO(piechotam) parameterize the layer
        self._model.avgpool.register_forward_hook(hook)

        with torch.no_grad():
            torch.cuda.empty_cache()
            _ = self._model(input_batch.float().to(self._device))

        return activations["output"][:, :, 0, 0].data

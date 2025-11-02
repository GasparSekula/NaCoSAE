import abc
from functools import wraps
from typing import Any, Callable, Literal

from absl import logging


def gpu_inference_wrapper(inference_func: Callable[[Any], Any]):
    """Sends model to GPU for inference and then sends it back to CPU."""

    @wraps(inference_func)
    def wrapper(self: Model, *args, **kwargs):
        self._send_to_device("cuda")
        try:
            result = inference_func(self, *args, **kwargs)
        finally:
            self._send_to_device("cpu")
        return result

    return wrapper


class Model(abc.ABC):
    def __init__(self, model_id: str, device: str) -> None:
        self._model_id = model_id
        self._device = device
        self._model = None
        self._load()

    @abc.abstractmethod
    def _load(self):
        """Loads the model resources."""
        pass

    def _send_to_device(self, device: Literal["cuda", "cpu"]):
        """Send model to device."""
        logging.info(
            "Sending model %s to device %s." % (self._model_id, device)
        )
        self._model.to(device)

    @property
    def model_id(self):
        return self._model_id

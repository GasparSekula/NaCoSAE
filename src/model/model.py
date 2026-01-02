import abc
from functools import wraps
from typing import Any, Callable, Literal
import warnings

from absl import logging


def gpu_inference_wrapper(inference_func: Callable[[Any], Any]):
    """Sends model to GPU for inference and then sends it back to CPU."""

    @wraps(inference_func)
    def wrapper(self: Model, *args, **kwargs):
        if self._model_swapping:
            self._send_to_device("cuda")
            result = inference_func(self, *args, **kwargs)
            self._send_to_device("cpu")
        else:
            result = inference_func(self, *args, **kwargs)

        return result

    return wrapper


class Model(abc.ABC):
    def __init__(
        self, model_id: str, device: str, model_swapping: bool, **load_kwargs
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._model_swapping = model_swapping
        self._model = self._load(**load_kwargs)

    @abc.abstractmethod
    def _load(self, **kwargs):
        """Loads the model resources."""
        pass

    def _send_to_device(self, device: Literal["cuda", "cpu"]):
        """
        Send model to device. CPU inference warnings are filtered as no
        inference on the CPU is ever made.
        """
        logging.info(
            "Sending model %s to device %s." % (self._model_id, device)
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*use another device for inference.*"
            )
            self._model.to(device)

    @property
    def model_id(self):
        return self._model_id

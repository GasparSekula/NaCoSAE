"""Abstract base class and utilities for model management.

This module provides a base Model class for managing different model types
and utilities for handling GPU memory optimization during inference.
"""

import abc
from functools import wraps
from typing import Any, Callable, Literal
import warnings

from absl import logging


def gpu_inference_wrapper(inference_func: Callable[[Any], Any]):
    """Decorator to manage GPU memory by swapping models between CPU and GPU.

    Sends the model to GPU before inference and returns it to CPU after,
    allowing efficient memory management. Only activates if model_swapping
    is enabled in the Model instance.

    Args:
        inference_func: The inference function to wrap.

    Returns:
        Wrapped function that manages device placement.
    """

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
    """Abstract base class for managing machine learning models.

    Provides common functionality for loading, device management, and
    model swapping between CPU and GPU for efficient memory usage.
    """

    def __init__(
        self, model_id: str, device: str, model_swapping: bool, **load_kwargs
    ) -> None:
        """Initialize the model.

        Args:
            model_id: Identifier or path for the model to load.
            device: Initial device to load model on ('cuda' or 'cpu').
            model_swapping: Whether to enable GPU/CPU swapping during inference.
            **load_kwargs: Additional keyword arguments passed to _load().
        """
        self._model_id = model_id
        self._device = device
        self._model_swapping = model_swapping
        self._model = self._load(**load_kwargs)

    @abc.abstractmethod
    def _load(self, **kwargs):
        """Load model resources from disk.

        Subclasses must implement this method to load and initialize the model
        with appropriate pretrained weights or architecture.

        Args:
            **kwargs: Model-specific loading arguments.

        Returns:
            Loaded model object.
        """
        pass

    def _send_to_device(self, device: Literal["cuda", "cpu"]):
        """Move model to specified device with warning suppression.

        Transfers the model to the specified device (GPU or CPU). CPU inference
        warnings are suppressed since the model is not meant to perform inference
        on CPU.

        Args:
            device: Target device ('cuda' for GPU or 'cpu' for CPU).
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
        """Get the model identifier.

        Returns:
            The model ID used for initialization.
        """
        return self._model_id

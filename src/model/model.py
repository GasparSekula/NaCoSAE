import abc

class Model(abc.ABC):
    def __init__(self, model_id: str, device: str) -> None:
        self._model_id = model_id
        self._device = device
    
    @abc.abstractmethod
    def _load(self):
        """Loads the model resources."""
        pass

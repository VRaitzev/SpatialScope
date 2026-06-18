from abc import ABC, abstractmethod
from typing import Any

class AbstractTrainPipeline(ABC):
    """
    [APPLICATION LAYER] Interface for the end-to-end machine learning training pipeline.

    This abstraction coordinates the orchestration of optimization loops, ensuring 
    a standardized contract for data management and model training across various
    multimodal experiments.
    """

    @abstractmethod
    def data_prepare(self, *args, **kwargs) -> Any:
        """
        Prepares and configures DataLoaders, datasets, or stream structures 
        required for the execution of the training cycle.
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """
        Orchestrates the global optimization loop, handling epoch iterations,
        forward/backward passes, and weight updates.
        """
        pass
from abc import ABC, abstractmethod
from app.core.schemes.Cells import ExpertModelManifest, WSICellFeatureRegistry
from app.core.schemes.Segmentation import SegmentationContainer

class AbstractExpertModel(ABC):
    """
    [MODEL LAYER] Interface for expert-level feature extraction and molecular profiling.

    This abstraction defines the contract for models that transform raw segmented 
    histological imagery into rich latent representations (embeddings) and 
    domain-specific features used for downstream gene expression prediction.
    """

    @property
    @abstractmethod
    def manifest(self) -> ExpertModelManifest:
        """
        Returns the model manifest, including metadata and supported feature definitions.
        """
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """
        Loads pre-trained model weights from the specified filesystem path.

        Args:
            path: Path to the serialized model state (e.g., .pth, .ckpt).
        """
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """
        Serializes current model weights to the specified filesystem path.

        Args:
            path: Destination path for model state persistence.
        """
        pass

    @abstractmethod
    def predict(self, cell_registry: SegmentationContainer) -> WSICellFeatureRegistry:
        """
        Performs the forward-pass inference for feature extraction.

        Takes a container of segmented cells, computes expert morphological 
        embeddings, and returns a structured registry populated with cell-level 
        feature passports.

        Args:
            cell_registry: Segmentation results containing geometry and centroid data.

        Returns:
            WSICellFeatureRegistry containing enriched cell representations.
        """
        pass
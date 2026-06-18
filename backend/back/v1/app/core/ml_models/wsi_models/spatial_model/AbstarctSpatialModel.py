from abc import ABC, abstractmethod
from typing import List
from app.core.schemes.Cells import (
    ExpertModelManifest, 
    WSICellFeatureRegistry, 
    RegionalSpatialSpot, 
    RegionalPredictionOutput
)

class AbstractSpatialSpotModel(ABC):
    """
    [MODEL LAYER] Interface for spatial gene expression prediction.

    This abstraction defines the final stage of the multimodal pipeline. It integrates 
    morphological cell-level embeddings with transcriptomic spot locations using 
    spatial aggregation (pooling) to predict gene expression profiles at high resolution.
    """

    @property
    @abstractmethod
    def manifest(self) -> ExpertModelManifest:
        """Returns the model manifest, specifying the target genes and model architecture."""
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """
        Loads the pre-trained spatial prediction model (e.g., GNN or Spatial Transformer) weights.

        Args:
            path: Path to the model state file.
        """
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """
        Serializes model parameters to disk.

        Args:
            path: Target storage path.
        """
        pass

    @abstractmethod
    def predict_spatial_expression(
        self, 
        patch_registry: WSICellFeatureRegistry, 
        regional_spots: List[RegionalSpatialSpot]
    ) -> RegionalPredictionOutput:
        """
        Computes spatial gene expression profiles for a given histological region.

        This method performs the aggregation of morphological features from 
        contained cells into local spatial spots to estimate target gene activity.

        Args:
            patch_registry: Enriched registry of cellular morphological features and embeddings.
            regional_spots: List of transcriptomic spots mapped to their constituent cells.

        Returns:
            RegionalPredictionOutput containing the predicted expression values (Y_hat).
        """
        pass
from abc import ABC, abstractmethod
from typing import List
from app.core.schemes.WSI import WSIContainer, WSIRegionContainer
from app.core.schemes.Segmentation import SegmentationContainer
from app.core.schemes.Cells import WSICellFeatureRegistry, RegionalPredictionOutput
from app.core.ml_models.wsi_models import (
    AbstractSegmentationModel, 
    AbstractExpertModel, 
    AbstractSpatialSpotModel
)

class AbstractInferencePipeline(ABC):
    """
    [APPLICATION LAYER] Interface for the end-to-end inference pipeline.

    Coordinates the sequential processing of histological data (clinical WSI), 
    orchestrating segmentation, feature extraction, and spatial prediction 
    to infer gene expression in the absence of transcriptomic ground truth.
    """

    def __init__(
        self,
        segmentation_model: AbstractSegmentationModel,
        expert_model: AbstractExpertModel,
        spatial_model: AbstractSpatialSpotModel
    ):
        """
        Initializes the pipeline components.

        Args:
            segmentation_model: The cellular segmentation model (e.g., StarDist).
            expert_model: The morphological feature/embedding extractor (e.g., DINOv2).
            spatial_model: The spatial GNN model for gene expression prediction.
        """
        self.segmentation_model = segmentation_model
        self.expert_model = expert_model
        self.spatial_model = spatial_model

    @abstractmethod
    def data_prepare(self, path: str) -> WSIContainer:
        """
        Initializes WSI metadata and spatial grid for clinical inference.

        Args:
            path: System path to the clinical WSI file.
        """
        pass

    @abstractmethod
    def patch_segmentation(self, wsi_patch: WSIRegionContainer) -> SegmentationContainer:
        """Stage 1: Detects cellular boundaries within a given WSI patch."""
        pass

    @abstractmethod
    def patch_feature_extraction(self, segments: SegmentationContainer) -> WSICellFeatureRegistry:
        """Stage 2: Generates morphological latent representations for segmented cells."""
        pass

    @abstractmethod
    def patch_predict(self, cells: WSICellFeatureRegistry) -> RegionalPredictionOutput:
        """
        Stage 3: Executes the spatial GNN forward-pass to infer gene expression.
        """
        pass

    @abstractmethod
    def predict(self, path: str) -> List[RegionalPredictionOutput]:
        """
        Orchestrates the global inference workflow across the entire WSI.

        Args:
            path: Path to the clinical sample file.

        Returns:
            A collection of regional expression predictions across the slide.
        """
        pass
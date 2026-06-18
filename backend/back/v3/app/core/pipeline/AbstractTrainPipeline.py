from abc import ABC, abstractmethod
from typing import List
from app.core.schemes.WSI import WSIContainer, WSIRegionContainer
from app.core.schemes.Segmentation import SegmentationContainer
from app.core.schemes.Cells import (
    WSICellFeatureRegistry, 
    RegionalPredictionOutput, 
    RegionalTargetOutput
)
from app.core.ml_models.wsi_models import (
    AbstractSegmentationModel, 
    AbstractExpertModel, 
    AbstractSpatialSpotModel
)

class AbstractTrainPipeline(ABC):
    """
    [APPLICATION LAYER] Interface for the end-to-end training pipeline.

    This abstraction coordinates the training process for the multimodal pipeline,
    linking cellular morphological features with transcriptomic ground truth to 
    optimize spatial gene expression prediction.
    """

    def __init__(
        self,
        segmentation_model: AbstractSegmentationModel,
        expert_model: AbstractExpertModel,
        spatial_model: AbstractSpatialSpotModel
    ):
        """
        Initializes the training pipeline components.

        Args:
            segmentation_model: The cellular segmentation model.
            expert_model: The morphological feature extractor.
            spatial_model: The spatial GNN model to be trained.
        """
        self.segmentation_model = segmentation_model
        self.expert_model = expert_model
        self.spatial_model = spatial_model

    @abstractmethod
    def data_prepare(self, path: str) -> WSIContainer:
        """
        Prepares WSI metadata and grid indices for training, including 
        loading transcriptomic ground truth.
        """
        pass

    @abstractmethod
    def patch_segmentation(self, wsi_patch: WSIRegionContainer) -> SegmentationContainer:
        """
        Stage 1: Performs nuclear segmentation on the training patch.
        """
        pass

    @abstractmethod
    def patch_feature_extraction(self, segments: SegmentationContainer) -> WSICellFeatureRegistry:
        """
        Stage 2: Extracts morphological latent representations for segmented cells.
        """
        pass

    @abstractmethod
    def patch_forward(
        self, 
        cells: WSICellFeatureRegistry, 
        target: RegionalTargetOutput
    ) -> RegionalPredictionOutput:
        """
        Stage 3: Performs the forward pass during training, computing predictions 
        against known transcriptomic targets for loss calculation.
        """
        pass

    @abstractmethod
    def train(self, path: str) -> None:
        """
        Orchestrates the global training loop over the provided dataset.

        Args:
            path: Path to the training dataset.
        """
        pass
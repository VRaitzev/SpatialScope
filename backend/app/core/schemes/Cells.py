from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union
from app.core.schemes.Segmentation import SegmentationCellInstance

class FeatureMeta(BaseModel):
    """Global schema definition for biological feature interpretation."""
    feature_name: str = Field(..., description="Technical identifier (e.g., 'nuclear_area').")
    description: str = Field(..., description="Biological context (e.g., 'Nucleus area').")
    unit: Optional[str] = Field(None, description="Measurement unit (e.g., 'µm²').")


class ExpertModelManifest(BaseModel):
    """Registry for expert model architecture and supported feature sets."""
    model_name: str = Field(..., description="Unique model identifier.")
    model_version: str = Field(..., description="Version of model weights/architecture.")
    model_description: str = Field(..., description="Biological objective of the model.")
    supported_features: List[FeatureMeta] = Field(
        ..., 
        description="Catalog of features extractable by this model."
    )


# =====================================================================
# CELL DATA CONTRACTS
# =====================================================================

class CellFeatureValue(BaseModel):
    """Encoded feature value mapped to a specific schema definition."""
    feature_name: str = Field(..., description="Link to FeatureMeta definition.")
    value: Union[float, int, str, List[float]] = Field(..., description="Calculated feature metric.")


class CellFeaturePassport(BaseModel):
    """
    Standardized cell-level representation.
    Encapsulates geometry and extracted morphological/feature profiles.
    """
    cell_id: str = Field(..., description="Unique cell identifier within the slide.")
    slide_id: str = Field(..., description="Parent WSI identifier.")
    
    segmentation_data: SegmentationCellInstance = Field(
        ..., 
        description="Segmentation mask, centroid, and bounding box data."
    )

    expert_features: Dict[str, List[CellFeatureValue]] = Field(
        default_factory=dict, 
        description="Map of expert model results: {model_name: [feature_values]}."
    )
    
    expert_embeddings: Dict[str, List[float]] = Field(
        default_factory=dict, 
        description="Latent feature vectors for downstream GNN inference."
    )


# =====================================================================
# REGISTRY AND PREDICTION CONTRACTS
# =====================================================================

class WSICellFeatureRegistry(BaseModel):
    """
    Top-level patch container.
    Integrates model manifests with localized cell-level data.
    """
    slide_id: str = Field(..., description="Histological slide identifier.")
    models_manifests: List[ExpertModelManifest] = Field(
        default_factory=list, 
        description="Manifests for all models applied to this slide."
    )
    x_wsi: float = Field(..., description="Global patch center X coordinate.")
    y_wsi: float = Field(..., description="Global patch center Y coordinate.")
    x_local: float = Field(..., description="Local patch-relative center X.")
    y_local: float = Field(..., description="Local patch-relative center Y.")
    cells: List[CellFeaturePassport] = Field(
        default_factory=list, 
        description="List of detected cellular passports."
    )


class SpotExpressionTarget(BaseModel):
    """Ground truth transcriptomic spot data for training pipelines."""
    spot_id: str = Field(..., description="Transcriptomic spot identifier.")
    x_wsi: float = Field(...)
    y_wsi: float = Field(...)
    x_local: float = Field(...)
    y_local: float = Field(...)
    target_expressions: Optional[Dict[str, float]] = Field(
        None, 
        description="Transcriptomic ground truth. None during inference."
    )


class RegionalTargetOutput(BaseModel):
    """Training-specific output container for regional patches."""
    patch_id: str = Field(...)
    predictions: List[SpotExpressionTarget] = Field(...)


class SpotExpressionPrediction(BaseModel):
    """Model inference result for a specific spatial spot."""
    spot_id: str = Field(..., description="Transcriptomic spot identifier.")
    predicted_expressions: Dict[str, float] = Field(
        ..., 
        description="Predicted gene expression (Y_hat)."
    )


class RegionalPredictionOutput(BaseModel):
    """[INFERENCE LAYER] Final prediction container for a regional patch."""
    patch_id: str = Field(...)
    predictions: List[SpotExpressionPrediction] = Field(...)

# Синтаксический мост для старых импортов в пространственных моделях
RegionalSpatialSpot = SpotExpressionTarget
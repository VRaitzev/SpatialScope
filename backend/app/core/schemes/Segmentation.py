from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Optional, List, Dict, Tuple, Union
import torch
from shapely.geometry import Polygon as ShapelyPolygon
from shapely import wkb

# =====================================================================
# MODEL MANIFESTS & METADATA
# =====================================================================

class SegmentationFeatureMeta(BaseModel):
    """Global schema definition for cellular feature interpretation."""
    feature_name: str = Field(..., description="Technical identifier (e.g., 'nuclear_area').")
    description: str = Field(..., description="Biological context (e.g., 'Area of the nucleus').")
    unit: Optional[str] = Field(None, description="Measurement unit (e.g., 'µm²').")


class SegmentationModelManifest(BaseModel):
    """Registry for segmentation model architecture and supported feature sets."""
    model_name: str = Field(..., description="Unique model identifier.")
    model_version: str = Field(..., description="Version of model weights/architecture.")
    model_description: str = Field(..., description="Biological objective of the model.")
    supported_features: List[SegmentationFeatureMeta] = Field(
        ..., 
        description="Catalog of features extractable by this model."
    )


# =====================================================================
# GEOMETRY & DATA STRUCTURES
# =====================================================================

class PolygonModel(BaseModel):
    """Handles polygon representation with seamless Shapely and WKB integration."""
    points: List[Tuple[float, float]]

    def as_shapely(self) -> ShapelyPolygon:
        """Returns a Shapely object for spatial computations."""
        return ShapelyPolygon(self.points)

    def to_wkb(self) -> bytes:
        """Serializes to WKB format for binary storage (e.g., Parquet/DB)."""
        return wkb.dumps(self.as_shapely())

    @classmethod
    def from_wkb(cls, data: bytes) -> 'PolygonModel':
        poly = wkb.loads(data)
        return cls.from_shapely(poly)

    @classmethod
    def from_shapely(cls, poly: ShapelyPolygon) -> 'PolygonModel':
        return cls(points=list(poly.exterior.coords))


class SegmentationFeatureValue(BaseModel):
    """Encoded feature value mapped to a specific schema definition."""
    feature_name: str = Field(..., description="Link to FeatureMeta identifier.")
    value: Union[float, int, str, List[float]]


# =====================================================================
# INPUT & OUTPUT CONTAINERS
# =====================================================================

class ImageMetadata(BaseModel):
    slide_id: str
    x: int
    y: int
    magnification: Optional[float] = None


class SegmentationInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_tensor: torch.Tensor
    metadata: Optional[ImageMetadata] = None

    @field_validator('image_tensor')
    @classmethod
    def validate_tensor_structure(cls, v: torch.Tensor) -> torch.Tensor:
        if v.numel() == 0:
            raise ValueError("Image tensor is empty.")
        if len(v.shape) != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got {len(v.shape)}D.")
        if v.shape[0] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {v.shape[0]}.")
        if v.dtype != torch.float32:
            raise ValueError("Expected torch.float32 dtype.")
        return v


class SegmentationCellInstance(BaseModel):
    """Represents a single detected cell instance."""
    polygon: PolygonModel
    bbox: Tuple[int, int, int, int]
    score: float = Field(ge=0.0, le=1.0)
    centroid: Optional[Tuple[float, float]] = None
    features: Dict[str, SegmentationFeatureValue] = Field(default_factory=dict)

    @model_validator(mode='after')
    def calculate_centroid_if_missing(self) -> 'SegmentationCellInstance':
        if self.centroid is None:
            poly = self.polygon.as_shapely()
            self.centroid = (float(poly.centroid.x), float(poly.centroid.y))
        return self


class SegmentationContainer(BaseModel):
    """Output container for total patch segmentation results."""
    instances: List[SegmentationCellInstance]
    image_shape: Tuple[int, int]
    model_name: Optional[str] = None
    version: Optional[str] = None

    def get_summary(self) -> dict:
        """Returns diagnostic statistics for the patch."""
        count = len(self.instances)
        avg_score = sum(c.score for c in self.instances) / count if count > 0 else 0
        return {
            "cell_count": count,
            "avg_score": avg_score,
            "model": f"{self.model_name}:{self.version}"
        }
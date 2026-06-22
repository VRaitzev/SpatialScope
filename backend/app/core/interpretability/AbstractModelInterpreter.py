from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureAttribution(BaseModel):
    feature_name: str
    importance: float
    description: Optional[str] = None


class CellAttributionMap(BaseModel):
    cell_id: str
    centroid: List[float]
    importance_score: float
    attributions: List[FeatureAttribution] = Field(default_factory=list)


class SpatialAttributionMap(BaseModel):
    width: int
    height: int
    values: List[List[float]] = Field(default_factory=list)


class InterpretationResult(BaseModel):
    model_type: str
    model_name: str
    summary: str
    cell_maps: List[CellAttributionMap] = Field(default_factory=list)
    spatial_map: Optional[SpatialAttributionMap] = None
    top_features: List[FeatureAttribution] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AbstractModelInterpreter(ABC):
    """Interface for post-hoc model interpretation and attribution maps."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @abstractmethod
    def interpret(
        self,
        *,
        patch_image=None,
        cell_registry=None,
        spatial_model=None,
        target_gene: Optional[str] = None,
        patch_id: str = "patch",
    ) -> InterpretationResult:
        pass

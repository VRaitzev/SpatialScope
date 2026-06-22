from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.ml_models.wsi_models.spatial_model.AbstarctSpatialModel import AbstractSpatialSpotModel


class PresetConfiguration(BaseModel):
    """Preset metadata for validation and frontend."""
    preset_id: str
    name: str
    description: str
    expert_ids: List[str] = Field(..., min_length=1)
    spatial_model_id: str
    composite_expert_key: Optional[str] = None


class AbstractModelRegistry(ABC):
    """Interface for dynamic model / preset management."""

    @abstractmethod
    def register_expert(self, expert_id: str, expert: AbstractExpertModel) -> None:
        pass

    @abstractmethod
    def register_spatial_model(self, model_id: str, model: AbstractSpatialSpotModel) -> None:
        pass

    @abstractmethod
    def create_preset(
        self,
        preset_id: str,
        name: str,
        description: str,
        expert_ids: List[str],
        spatial_model_id: str,
    ) -> PresetConfiguration:
        pass

    @abstractmethod
    def set_active_preset(self, preset_id: str) -> None:
        pass

    @abstractmethod
    def get_active_models(self) -> Tuple[AbstractExpertModel, AbstractSpatialSpotModel]:
        pass

    @abstractmethod
    def get_models_for_preset(
        self, preset_id: str
    ) -> Tuple[AbstractExpertModel, AbstractSpatialSpotModel]:
        pass

    @abstractmethod
    def get_preset(self, preset_id: str) -> PresetConfiguration:
        pass

    @abstractmethod
    def get_available_presets(self) -> Dict[str, PresetConfiguration]:
        pass

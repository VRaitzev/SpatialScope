from typing import Dict, List, Tuple, Optional

from app.core.registry.AbstractModelRegistry import AbstractModelRegistry, PresetConfiguration
from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.ml_models.wsi_models.expert_models.CompositeExpertAdapter import CompositeExpertAdapter
from app.core.ml_models.wsi_models.spatial_model.AbstarctSpatialModel import AbstractSpatialSpotModel


class ModelPresetRegistry(AbstractModelRegistry):
    """In-memory registry of experts, spatial models and presets."""

    def __init__(self):
        self._experts: Dict[str, AbstractExpertModel] = {}
        self._spatial_models: Dict[str, AbstractSpatialSpotModel] = {}
        self._presets: Dict[str, PresetConfiguration] = {}
        self._active_preset_id: Optional[str] = None

    def register_expert(self, expert_id: str, expert: AbstractExpertModel) -> None:
        if not isinstance(expert, AbstractExpertModel):
            raise TypeError(f"Expected AbstractExpertModel, got {type(expert)}")
        self._experts[expert_id] = expert

    def register_spatial_model(self, model_id: str, model: AbstractSpatialSpotModel) -> None:
        if not isinstance(model, AbstractSpatialSpotModel):
            raise TypeError(f"Expected AbstractSpatialSpotModel, got {type(model)}")
        self._spatial_models[model_id] = model

    def get_expert(self, expert_id: str) -> AbstractExpertModel:
        if expert_id not in self._experts:
            raise KeyError(f"Expert '{expert_id}' not found")
        return self._experts[expert_id]

    def get_spatial_model(self, model_id: str) -> AbstractSpatialSpotModel:
        if model_id not in self._spatial_models:
            raise KeyError(f"Spatial model '{model_id}' not found")
        return self._spatial_models[model_id]

    def _build_composite_key(self, preset_id: str) -> str:
        return f"Composite_{preset_id}"

    def create_preset(
        self,
        preset_id: str,
        name: str,
        description: str,
        expert_ids: List[str],
        spatial_model_id: str,
    ) -> PresetConfiguration:
        for expert_id in expert_ids:
            if expert_id not in self._experts:
                raise KeyError(f"Expert '{expert_id}' not found")
        if spatial_model_id not in self._spatial_models:
            raise KeyError(f"Spatial model '{spatial_model_id}' not found")

        composite_key = None
        if len(expert_ids) > 1:
            composite_key = self._build_composite_key(preset_id)
            experts = [self._experts[eid] for eid in expert_ids]
            self._experts[composite_key] = CompositeExpertAdapter(
                experts=experts,
                composite_key=composite_key,
            )

        preset = PresetConfiguration(
            preset_id=preset_id,
            name=name,
            description=description,
            expert_ids=expert_ids,
            spatial_model_id=spatial_model_id,
            composite_expert_key=composite_key,
        )
        self._presets[preset_id] = preset

        if self._active_preset_id is None:
            self._active_preset_id = preset_id

        return preset

    def set_active_preset(self, preset_id: str) -> None:
        if preset_id not in self._presets:
            raise KeyError(f"Preset '{preset_id}' not found")
        self._active_preset_id = preset_id

    def get_preset(self, preset_id: str) -> PresetConfiguration:
        if preset_id not in self._presets:
            raise KeyError(f"Preset '{preset_id}' not found")
        return self._presets[preset_id]

    def _resolve_expert_key(self, preset: PresetConfiguration) -> str:
        if preset.composite_expert_key:
            return preset.composite_expert_key
        return preset.expert_ids[0]

    def get_models_for_preset(
        self, preset_id: str
    ) -> Tuple[AbstractExpertModel, AbstractSpatialSpotModel]:
        preset = self.get_preset(preset_id)
        expert_key = self._resolve_expert_key(preset)
        expert = self._experts[expert_key]
        spatial = self._spatial_models[preset.spatial_model_id]
        if preset.composite_expert_key:
            spatial.expert_key = preset.composite_expert_key
        else:
            spatial.expert_key = getattr(expert, "EXPERT_KEY", preset.expert_ids[0])
        return expert, spatial

    def get_active_models(self) -> Tuple[AbstractExpertModel, AbstractSpatialSpotModel]:
        if not self._active_preset_id:
            raise RuntimeError("No active preset selected")
        return self.get_models_for_preset(self._active_preset_id)

    def get_available_presets(self) -> Dict[str, PresetConfiguration]:
        return self._presets

    def get_weights_path(self, preset_id: str) -> str:
        preset = self.get_preset(preset_id)
        return f"./data/checkpoints/{preset_id}/{preset.spatial_model_id}_best.pt"

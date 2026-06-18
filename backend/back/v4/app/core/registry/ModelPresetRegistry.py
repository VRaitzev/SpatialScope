from app.core.registry.AbstractModelRegistry import AbstractModelRegistry, PresetConfiguration
from typing import Dict, Tuple, Optional, Any
from pydantic import BaseModel

# Импортируем твои абстрактные классы моделей
from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.ml_models.wsi_models.spatial_model.AbstarctSpatialModel import AbstractSpatialSpotModel

class ModelPresetRegistry(AbstractModelRegistry):
    """
    [PRODUCTION LAYER] Конкретная реализация реестра.
    Хранит инстансы живых моделей в памяти, управляет конфигурациями пресетов.
    """

    def __init__(self):
        self._experts: Dict[str, AbstractExpertModel] = {}
        self._spatial_models: Dict[str, AbstractSpatialSpotModel] = {}
        self._presets: Dict[str, PresetConfiguration] = {}
        self._active_preset_id: Optional[str] = None

    def register_expert(self, expert_id: str, expert: AbstractExpertModel) -> None:
        if not isinstance(expert, AbstractExpertModel):
            raise TypeError(f"Модель должна наследоваться от AbstractExpertModel. Получен: {type(expert)}")
        self._experts[expert_id] = expert
        print(f"🔹 [Registry] Эксперт '{expert_id}' успешно зарегистрирован. (Manifest: {expert.manifest.model_name})")

    def register_spatial_model(self, model_id: str, model: AbstractSpatialSpotModel) -> None:
        if not isinstance(model, AbstractSpatialSpotModel):
            raise TypeError(f"Модель должна наследоваться от AbstractSpatialSpotModel. Получен: {type(model)}")
        self._spatial_models[model_id] = model
        print(f"🧠 [Registry] Пространственная модель '{model_id}' успешно зарегистрирована.")

    def create_preset(self, preset_id: str, name: str, description: str, expert_id: str, spatial_model_id: str) -> PresetConfiguration:
        if expert_id not in self._experts:
            raise KeyError(f"Невозможно создать пресет: Эксперт '{expert_id}' не найден в реестре.")
        if spatial_model_id not in self._spatial_models:
            raise KeyError(f"Невозможно создать пресет: Пространственная модель '{spatial_model_id}' не найдена в реестре.")

        preset = PresetConfiguration(
            preset_id=preset_id,
            name=name,
            description=description,
            expert_id=expert_id,
            spatial_model_id=spatial_model_id
        )
        self._presets[preset_id] = preset
        
        # Если это самый первый пресет в системе, делаем его активным автоматически
        if self._active_preset_id is None:
            self._active_preset_id = preset_id
            
        print(f"🎯 [Registry] Создан пресет конфигурации: [{name}] (ID: {preset_id})")
        return preset

    def set_active_preset(self, preset_id: str) -> None:
        if preset_id not in self._presets:
            raise KeyError(f"Пресет '{preset_id}' отсутствует в системе.")
        self._active_preset_id = preset_id
        print(f"⚙️ [Registry] Активный пресет переключен на: {self._presets[preset_id].name}")

    def get_active_models(self) -> Tuple[AbstractExpertModel, AbstractSpatialSpotModel]:
        if not self._active_preset_id:
            raise RuntimeError("Запрос моделей отклонен: в системе не выбран активный пресет.")
            
        config = self._presets[self._active_preset_id]
        expert_instance = self._experts[config.expert_id]
        spatial_instance = self._spatial_models[config.spatial_model_id]
        
        # Динамически прокидываем в адаптер GNN ключ эксперта, чтобы граф собрался по нужным фичам!
        # Твой SpatialGNNAdapter ждет self.expert_key для чтения эмбеддингов
        spatial_instance.expert_key = expert_instance.EXPERT_KEY
        
        return expert_instance, spatial_instance

    def get_available_presets(self) -> Dict[str, PresetConfiguration]:
        return self._presets
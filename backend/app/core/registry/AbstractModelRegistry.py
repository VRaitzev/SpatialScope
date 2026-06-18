from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
from pydantic import BaseModel

# Импортируем твои абстрактные классы моделей
from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.ml_models.wsi_models.spatial_model.AbstarctSpatialModel import AbstractSpatialSpotModel

class PresetConfiguration(BaseModel):
    """Схема метаданных пресета для валидации и фронтенда"""
    preset_id: str
    name: str
    description: str
    expert_id: str
    spatial_model_id: str


class AbstractModelRegistry(ABC):
    """
    [REGISTRY LAYER] Интерфейс для динамического управления моделями, весами и пресетами.
    """

    @abstractmethod
    def register_expert(self, expert_id: str, expert: AbstractExpertModel) -> None:
        """Регистрация модели-эксперта первого уровня (Морфология, EffNet, Радиомика)."""
        pass

    @abstractmethod
    def register_spatial_model(self, model_id: str, model: AbstractSpatialSpotModel) -> None:
        """Регистрация предсказательной пространственной модели (GNN, Transformer)."""
        pass

    @abstractmethod
    def create_preset(self, preset_id: str, name: str, description: str, expert_id: str, spatial_model_id: str) -> PresetConfiguration:
        """Создание связки Эксперт + GNN для быстрого переключения пользователем."""
        pass

    @abstractmethod
    def set_active_preset(self, preset_id: str) -> None:
        """Глобальная установка текущего рабочего пресета."""
        pass

    @abstractmethod
    def get_active_models(self) -> Tuple[AbstractExpertModel, AbstractSpatialSpotModel]:
        """Возвращает кортеж инстансов моделей из активного пресета для инференса."""
        pass

    @abstractmethod
    def get_available_presets(self) -> Dict[str, PresetConfiguration]:
        """Возвращает список всех доступных пресетов для вывода на фронтенд."""
        pass

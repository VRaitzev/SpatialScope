from abc import ABC, abstractmethod
from app.core.schemes.WSI import WSIRegionContainer
from app.core.schemes.Segmentation import SegmentationContainer, SegmentationModelManifest

class AbstractSegmentationModel(ABC):
    """
    [MODEL LAYER] Интерфейс модели сегментации клеток.
    Управляет жизненным циклом весов сети и инференсом на патчах ткани.
    """
    
    @property
    @abstractmethod
    def manifest(self) -> SegmentationModelManifest:
        """
        Возвращает манифест модели (имя, версию и список фичей, которые она считает).
        Пайплайн вызовет это свойство ОДИН раз, чтобы положить манифест в глобальный реестр.
        """
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Загрузка весов модели (.pth, .onnx, .bin) в память устройства"""
        pass
        
    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Сохранение текущего состояния весов модели на диск"""
        pass

    @abstractmethod
    def predict(self, patch: WSIRegionContainer) -> SegmentationContainer:
        """Основной forward-pass инференса сегментации клеток"""
        pass
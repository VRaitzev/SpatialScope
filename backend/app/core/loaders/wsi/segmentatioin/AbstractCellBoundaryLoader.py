"""
app/core/loaders/base_boundary_loader.py
Интерфейс загрузчика пространственных контуров клеток.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from app.core.schemes.WSI import WSIRegionContainer
from app.core.schemes.Segmentation import SegmentationContainer, SegmentationModelManifest

class AbstractCellBoundaryLoader(ABC):
    """
    [DATA LAYER / ADAPTER] Интерфейс загрузчика пространственных границ клеток.
    Отвечает за извлечение полигонов клеток и сопутствующих омиксных фичей из файлов данных.
    """
    
    @property
    @abstractmethod
    def manifest(self) -> SegmentationModelManifest:
        """
        Возвращает манифест загрузчика (какой прибор/пайплайн сгенерировал данные 
        и какие фичи экспрессии доступны).
        """
        pass

    @abstractmethod
    def load_boundaries(
        self, 
        patch: WSIRegionContainer, 
        registration_service: Optional[Any] = None
    ) -> SegmentationContainer:
        """
        Основной метод извлечения пространственных данных для указанного региона ткани.
        
        Args:
            patch (WSIRegionContainer): Координатный квадрат (тайл) на гистологическом слайде.
            registration_service (Optional[Any]): Сервис нелинейной регистрации (например, ANTsPy).
                Если передан, координаты полигонов и центроидов автоматически деформируются 
                из нативного пространства прибора в целевое пространство слайда H&E.
        """
        pass
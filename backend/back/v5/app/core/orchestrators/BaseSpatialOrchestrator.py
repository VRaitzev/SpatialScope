from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


class BaseSpatialOrchestrator(ABC):
    """
    Абстрактный интерфейс оркестратора (Фасада) информационной системы.
    Связывает веб-слой (FastAPI) с ML-движком и хранилищем данных.
    """

    @abstractmethod
    def register_new_slide(self, slide_path: str, cells_parquet_path: str) -> str:
        """
        Регистрирует новое WSI-стекло в системе, считывает его метаданные 
        и привязывает файл сегментации клеток. Возвращает уникальный slide_id.
        """
        pass

    @abstractmethod
    def get_slide_metadata(self, slide_id: str) -> Dict[str, Any]:
        """
        Возвращает метаданные слайда (размеры, количество клеток) для интерфейса.
        """
        pass

    @abstractmethod
    def get_patch_for_visualization(
        self, slide_id: str, x: int, y: int, patch_size: int
    ) -> Dict[str, Any]:
        """
        Вырезает патч для фронтенда: отдает координаты клеток, 
        границы (полигоны) и предсказанные экспрессии генов для интерактивной карты.
        """
        pass

    @abstractmethod
    def start_training_pipeline(
        self, expert_mode: str, epochs: int, train_patches_count: int
    ) -> str:
        """
        Инициализирует асинхронную задачу на обучение GNN модели.
        Возвращает job_id (ID задачи) для трекинга прогресса.
        """
        pass

    @abstractmethod
    def get_pipeline_status(self, job_id: str) -> Dict[str, Any]:
        """
        Возвращает текущий статус обучения (эпоха, текущий loss, статус: running/success/failed).
        Используется исследователем для вывода графиков на фронте.
        """
        pass

    @abstractmethod
    def run_inference_on_patch(
        self, slide_id: str, x: int, y: int, patch_size: int, weights_path: str
    ) -> Dict[str, Any]:
        """
        Запускает инференс (предсказание экспрессии) для конкретного региона 
        без переобучения модели (основной рабочий инструмент Врача).
        """
        pass
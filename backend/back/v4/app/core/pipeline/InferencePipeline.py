import torch
from typing import List, Iterator, Optional
from tqdm import tqdm

from app.core.schemes.WSI import WSIContainer, WSIRegionContainer
from app.core.schemes.Segmentation import SegmentationContainer
from app.core.schemes.Cells import WSICellFeatureRegistry
from app.core.schemes.RNA import RegionalPredictionOutput
from app.core.pipeline.AbstractInferencePipeline import AbstractInferencePipeline

class SpatialInferencePipeline(AbstractInferencePipeline):
    def __init__(
        self,
        wsi_loader,
        segmentation_model,
        expert_model,      # Наш GeometricMorphologyExpert
        spatial_model,     # Наш SpatialGNNAdapter
        patch_size: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.wsi_loader = wsi_loader
        self.segmentation_model = segmentation_model
        
        # Защита: геометрический эксперт детерминирован и не имеет весов PyTorch,
        # поэтому переносим на девайс только если у модели есть соответствующий метод.
        self.expert_model = expert_model
        if hasattr(self.expert_model, "to"):
            self.expert_model = self.expert_model.to(device)
        if hasattr(self.expert_model, "eval"):
            self.expert_model.eval()
            
        # Адаптер графовой сети
        self.spatial_model = spatial_model
        if hasattr(self.spatial_model, "model") and hasattr(self.spatial_model.model, "eval"):
            self.spatial_model.model.eval()
        elif hasattr(self.spatial_model, "eval"):
            self.spatial_model.eval()
        
        self.patch_size = patch_size
        self.device = device

    def data_prepare(self, path: str) -> WSIContainer:
        """Загружает метаданные слайда без загрузки пикселей в RAM."""
        return self.wsi_loader.load_slide_metadata(path)

    def _generate_patches(self, wsi: WSIContainer) -> Iterator[WSIRegionContainer]:
        """Генератор патчей для экономии памяти (Tiling)."""
        width, height = wsi.dimensions_level_0
        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):
                yield self.wsi_loader.load_region(
                    container=wsi,
                    x=x,
                    y=y,
                    width=self.patch_size,
                    height=self.patch_size,
                    level=0
                )

    @torch.no_grad()
    def patch_segmentation(self, wsi_patch: WSIRegionContainer) -> SegmentationContainer:
        """Шаг 1: Сегментация ядер/клеток (например, через StarDist)."""
        return self.segmentation_model.predict(wsi_patch)

    @torch.no_grad()
    def patch_feature_extraction(
        self, 
        segments: SegmentationContainer, 
        wsi_patch: WSIRegionContainer
    ) -> WSICellFeatureRegistry:
        """Шаг 2: Извлечение признаков клеток."""
        # АДАПТАЦИЯ: Больше не переносим image_tensor на GPU и не передаем его.
        # GeometricMorphologyExpert берет только геометрические контуры (segments).
        return self.expert_model.predict(segments)

    @torch.no_grad()
    def patch_predict(
        self, 
        cells: WSICellFeatureRegistry, 
        spots: List
    ) -> RegionalPredictionOutput:
        """Шаг 3: Графовый инференс и маппинг экспрессии на ST-споты."""
        # Вызывает метод SpatialGNNAdapter.predict_spatial_expression
        return self.spatial_model.predict_spatial_expression(cells, spots)

    def predict(self, path: str, spots_locations: Optional[List] = None) -> List[RegionalPredictionOutput]:
        """Полный цикл предсказания для целого гистологического слайда."""
        wsi_container = self.data_prepare(path)
        all_results = []

        # Покомпонентно обрабатываем весь слайд через тайлинг
        for patch in tqdm(self._generate_patches(wsi_container), desc="Processing WSI Patches"):
            
            # 1. Запуск сегментации (на выходе полигоны и центроиды)
            segments = self.patch_segmentation(patch)
            
            # Если в этом регионе слайда нет ткани или клеток — пропускаем его
            if not segments.instances:
                continue

            # 2. Быстрый расчет морфологии и локальной плотности на CPU
            cell_features = self.patch_feature_extraction(segments, patch)

            # 3. Фильтрация ST-спотов, попавших в координаты текущего патча
            patch_spots = self._filter_spots_for_patch(spots_locations, patch) if spots_locations else []

            # 4. Построение графа Делоне и предсказание профиля экспрессии через GATv2
            patch_predictions = self.patch_predict(cell_features, patch_spots)
            all_results.append(patch_predictions)

        return all_results

    def _filter_spots_for_patch(self, spots: List, patch: WSIRegionContainer) -> List:
        """Выбирает споты Visium/Xenium, попадающие в границы текущего патча."""
        minx, miny = patch.x_wsi, patch.y_wsi
        maxx, maxy = minx + patch.width_wsi, miny + patch.height_wsi
        
        return [
            s for s in spots 
            if minx <= s.x_wsi <= maxx and miny <= s.y_wsi <= maxy
        ]
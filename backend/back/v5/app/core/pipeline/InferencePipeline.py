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
        expert_model,
        spatial_model,
        patch_size: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        slide_id: str = "unknown_slide"  # <- НОВЫЙ ПАРАМЕТР
    ):
        self.wsi_loader = wsi_loader
        self.segmentation_model = segmentation_model
        self.expert_model = expert_model
        self.spatial_model = spatial_model
        self.patch_size = patch_size
        self.device = device
        self.slide_id = slide_id
        
        # Переносим на GPU если нужно
        if hasattr(self.expert_model, "to"):
            self.expert_model = self.expert_model.to(device)
        if hasattr(self.expert_model, "eval"):
            self.expert_model.eval()
            
        if hasattr(self.spatial_model, "model") and hasattr(self.spatial_model.model, "eval"):
            self.spatial_model.model.eval()
        elif hasattr(self.spatial_model, "eval"):
            self.spatial_model.eval()

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
        """Шаг 1: Сегментация ядер/клеток."""
        return self.segmentation_model.predict(wsi_patch)

    @torch.no_grad()
    def patch_feature_extraction(
        self, 
        segments: SegmentationContainer, 
        wsi_patch: WSIRegionContainer,
        patch_id: str
    ) -> WSICellFeatureRegistry:
        """
        Шаг 2: Извлечение признаков клеток.
        ИСПРАВЛЕНО: теперь передаются все необходимые параметры.
        """
        # Проверяем, нужна ли эксперту картинка
        expert_key = getattr(self.expert_model, 'EXPERT_KEY', '')
        
        if "EfficientNet" in expert_key or "Radiomics" in expert_key:
            # Для экспертов, которым нужно изображение
            if hasattr(wsi_patch, 'image_tensor'):
                patch_tensor = wsi_patch.image_tensor
            elif hasattr(wsi_patch, 'image'):
                import numpy as np
                patch_tensor = torch.from_numpy(wsi_patch.image).permute(2, 0, 1) if hasattr(wsi_patch.image, 'shape') else wsi_patch.image
            else:
                raise AttributeError("patch_region не содержит изображения для эксперта")
            
            return self.expert_model.predict(
                patch_image=patch_tensor,
                cell_registry=segments,
                patch_id=patch_id
            )
        else:
            # Для геометрического эксперта
            return self.expert_model.predict(
                cell_registry=segments,
                patch_id=patch_id,
                slide_id=self.slide_id
            )

    @torch.no_grad()
    def patch_predict(
        self, 
        cells: WSICellFeatureRegistry, 
        spots: List,
        patch_id: str
    ) -> RegionalPredictionOutput:
        """
        Шаг 3: Графовый инференс.
        ИСПРАВЛЕНО: передаём patch_id.
        """
        # Убеждаемся, что у spatial_model есть expert_key
        if not hasattr(self.spatial_model, 'expert_key'):
            if hasattr(self.expert_model, 'EXPERT_KEY'):
                self.spatial_model.expert_key = self.expert_model.EXPERT_KEY
            else:
                self.spatial_model.expert_key = "EfficientNetExpert"
        
        return self.spatial_model.predict_spatial_expression(
            patch_registry=cells,
            regional_spots=spots,
            patch_id=patch_id
        )

    def predict(self, path: str, spots_locations: Optional[List] = None) -> List[RegionalPredictionOutput]:
        """Полный цикл предсказания для целого гистологического слайда."""
        wsi_container = self.data_prepare(path)
        all_results = []

        # Счетчик патчей для уникальных ID
        patch_counter = 0
        
        for patch in tqdm(self._generate_patches(wsi_container), desc="Processing WSI Patches"):
            patch_id = f"patch_{self.slide_id}_{patch_counter}"
            patch_counter += 1
            
            # 1. Сегментация
            segments = self.patch_segmentation(patch)
            
            if not segments.instances:
                continue

            # 2. Извлечение признаков (ИСПРАВЛЕНО: передаём patch и patch_id)
            cell_features = self.patch_feature_extraction(segments, patch, patch_id)

            # 3. Фильтрация ST-спотов
            patch_spots = self._filter_spots_for_patch(spots_locations, patch) if spots_locations else []

            # 4. Предсказание (ИСПРАВЛЕНО: передаём patch_id)
            patch_predictions = self.patch_predict(cell_features, patch_spots, patch_id)
            all_results.append(patch_predictions)

        return all_results

    def _filter_spots_for_patch(self, spots: List, patch: WSIRegionContainer) -> List:
        """Выбирает споты, попадающие в границы текущего патча."""
        minx, miny = patch.x_wsi, patch.y_wsi
        maxx, maxy = minx + patch.width_wsi, miny + patch.height_wsi
        
        return [
            s for s in spots 
            if minx <= s.x_wsi <= maxx and miny <= s.y_wsi <= maxy
        ]
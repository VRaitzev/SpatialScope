from typing import Optional
from app.core.schemes.Segmentation import (
    SegmentationModelManifest, 
    SegmentationFeatureMeta, 
    SegmentationContainer, 
    SegmentationCellInstance, 
    SegmentationFeatureValue
)
from app.core.ml_models.wsi_models.segmentations_models.AbstractSegmentationModel import AbstractSegmentationModel
from app.core.schemes.WSI import WSIRegionContainer

# Импортируем наши новые SpatialData загрузчики
from .XeniumSpatialDataLoader import XeniumSpatialDataLoader
from .XeniumSpatialDataGeneLoader import XeniumSpatialDataGeneLoader

class XeniumSpatialSegmentationAdapter(AbstractSegmentationModel):
    def __init__(self, spatial_loader: XeniumSpatialDataLoader, gene_loader: XeniumSpatialDataGeneLoader):
        """
        Адаптер, который объединяет геометрические данные (полигоны) и молекулярный профиль (гены),
        извлеченные из единого контейнера SpatialData.
        """
        self.loader = spatial_loader
        self.gene_loader = gene_loader

    @property
    def manifest(self) -> SegmentationModelManifest:
        """
        Манифест модели, декларирующий, какие именно фичи клеток мы отдаем наружу.
        """
        return SegmentationModelManifest(
            model_name="Xenium_SpatialData_Native",
            model_version="2.0",
            model_description="Native Xenium cell boundaries + Gene Panel Expression via scverse SpatialData",
            supported_features=[
                SegmentationFeatureMeta(
                    feature_name="xenium_panel_expression",
                    description=f"Vector of {len(self.gene_loader.get_all_gene_names())} panel genes",
                    unit="counts"
                )
            ]
        )

    def predict(self, patch: WSIRegionContainer) -> SegmentationContainer:
        """
        Собирает все клетки в границах патча и обогащает их векторами экспрессии генов.
        """
        # 1. Вырезаем паспорта клеток, попавших в координаты текущего патча
        cells_registry = self.loader.get_cells_in_region(
            patch.x_wsi, patch.y_wsi, patch.width_wsi, patch.height_wsi
        )
        
        instances = []
        skipped = 0
        
        # 2. Итерируемся по клеткам и подтягиваем для каждой её генный профиль
        for cell in cells_registry:
            gene_vector = self.gene_loader.get_gene_vector(cell.cell_id)
            
            # Если клетка есть на изображении, но почему-то отсутствует в таблице экспрессии
            if gene_vector is None:
                skipped += 1
                continue
            
            # Упаковываем вектор генов в Pydantic-схему фичей
            feat_value = SegmentationFeatureValue(
                feature_name="xenium_panel_expression",
                value=gene_vector.tolist()  # Конвертируем в list[float] для валидации Pydantic
            )
            
            # Собираем инстанс клетки со всей геометрией и фичами
            instances.append(SegmentationCellInstance(
                polygon=cell.segmentation_data.polygon,
                bbox=cell.segmentation_data.bbox,
                score=cell.segmentation_data.score,
                centroid=cell.segmentation_data.centroid,
                features={"xenium_panel": feat_value}
            ))
        
        if skipped > 0:
            print(f"⚠️ Предупреждение: пропущено клеток без вектора экспрессии: {skipped}")
            
        print(f"✅ Адаптер: упаковано {len(instances)} клеток для текущего патча.")
        
        return SegmentationContainer(
            instances=instances,
            image_shape=(patch.width_wsi, patch.height_wsi),
            model_name=self.manifest.model_name,
            version=self.manifest.model_version
        )
        
    def load_weights(self, path: str): 
        """Метод абстрактного класса, для нативного адаптера веса не требуются"""
        pass
        
    def save_weights(self, path: str): 
        pass
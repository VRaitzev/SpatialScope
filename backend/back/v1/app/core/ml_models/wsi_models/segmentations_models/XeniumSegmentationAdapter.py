from app.core.schemes.Segmentation import SegmentationModelManifest, SegmentationFeatureMeta
from app.core.ml_models.wsi_models.segmentations_models.AbstractSegmentationModel import AbstractSegmentationModel
from app.core.loaders.wsi.image import XeniumWSILoader
from app.core.loaders.rna import XeniumGenePanelLoader
# Плюс не забудь импортировать другие важные компоненты

from app.core.schemes.Segmentation import SegmentationContainer, SegmentationCellInstance, SegmentationFeatureValue
from app.core.schemes.WSI import WSIRegionContainer

class XeniumSegmentationAdapter(AbstractSegmentationModel):
    def __init__(self, xenium_loader: XeniumWSILoader, gene_loader: XeniumGenePanelLoader):
        self.loader = xenium_loader
        self.gene_loader = gene_loader

    @property
    def manifest(self) -> SegmentationModelManifest:
        # Теперь мы честно указываем, что модель (адаптер) отдает признаки экспрессии
        return SegmentationModelManifest(
            model_name="Xenium_Onboard_Native",
            model_version="1.0",
            model_description="Native Xenium cell boundaries + Gene Panel Expression",
            supported_features=[
                SegmentationFeatureMeta(
                    feature_name="xenium_panel_expression",
                    description="Vector of 313 panel genes",
                    unit="counts"
                )
            ]
        )

    def predict(self, patch: WSIRegionContainer) -> SegmentationContainer:
            cells_registry = self.loader.get_cells_in_region(
                patch.x_wsi, patch.y_wsi, patch.width_wsi, patch.height_wsi
            )
            if cells_registry:
                sample_cell = cells_registry[0]
                print(f"🔍 Пример cell_id из get_cells_in_region: {repr(sample_cell.cell_id)}")
                print(f"Тип: {type(sample_cell.cell_id)}")
                
                # Пробуем вручную
                test_vec = self.gene_loader.get_gene_vector(sample_cell.cell_id)
                print(f"Результат get_gene_vector для sample: {'Найден' if test_vec is not None else 'None'}")
            instances = []
            skipped = 0
            
            for cell in cells_registry:
                gene_vector = self.gene_loader.get_gene_vector(cell.cell_id)
                
                if gene_vector is None:
                    skipped += 1
                    continue
                
                feat_value = SegmentationFeatureValue(
                    feature_name="xenium_panel_expression",
                    value=gene_vector.tolist()   # list[float] — самый безопасный вариант для pydantic
                )
                
                instances.append(SegmentationCellInstance(
                    polygon=cell.segmentation_data.polygon,
                    bbox=cell.segmentation_data.bbox,
                    score=cell.segmentation_data.score,
                    centroid=cell.segmentation_data.centroid,
                    features={"xenium_panel": feat_value}   # или "xenium_panel_expression"
                ))
            
            if skipped > 0:
                print(f"⚠️ Пропущено клеток без вектора экспрессии: {skipped}")
            
            print(f"✅ Создано {len(instances)} инстансов клеток с экспрессией")
            
            return SegmentationContainer(
                instances=instances,
                image_shape=(patch.width_wsi, patch.height_wsi),
                model_name=self.manifest.model_name,
                version=self.manifest.model_version
            )
        
    def load_weights(self, path: str): pass
    def save_weights(self, path: str): pass
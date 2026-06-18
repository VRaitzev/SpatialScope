import os
import numpy as np
import torch
import geopandas as gpd
import spatialdata as sd
from spatialdata.queries import bounding_box_query
from typing import List

from app.core.loaders.wsi.image.AbstractWSILoader import AbstractWSILoader
from app.core.schemes.WSI import WSIContainer, WSIRegionContainer, WSILevelMetadata
from app.core.schemes.Cells import CellFeaturePassport
from app.core.schemes.Segmentation import SegmentationCellInstance, PolygonModel

class XeniumSpatialDataLoader(AbstractWSILoader):
    def __init__(self, zarr_path: str, patch_margin: int = 15):
        """
        Инициализация загрузчика через контейнер SpatialData (.zarr).
        """
        self.patch_margin = patch_margin
        
        print(f"📦 Загрузка SpatialData контейнера: {zarr_path}")
        self.sdata = sd.read_zarr(zarr_path)
        
        # Автоматически определяем ключи для изображений и полигонов клеток
        self.image_key = list(self.sdata.images.keys())[0]      # Обычно 'he_image'
        self.shapes_key = list(self.sdata.shapes.keys())[0]    # Обычно 'cell_boundaries'
        
        # Извлекаем полигоны (это готовый GeoDataFrame)
        self.boundaries_gdf = self.sdata.shapes[self.shapes_key]
        
        # Задаем базовое имя координатной системы (в SpatialData Xenium это обычно 'global')
        self.coordinate_system = "global"
        
        # Сохраняем mpp для обратной совместимости с твоими контрактами схем
        self.mpp = 0.212  
        
        print(f"✅ Успешно загружено. Найдено полигонов клеток: {len(self.boundaries_gdf)}")

    def load_slide_metadata(self, file_path: str) -> WSIContainer:
        """
        Собирает метаданные слайда. Физические размеры берем напрямую из SpatialImage.
        """
        image_obj = self.sdata.images[self.image_key]
        
        # В SpatialData оси могут быть упорядочены как (c, y, x)
        h = image_obj.sizes['y']
        w = image_obj.sizes['x']
        
        return WSIContainer(
            slide_id="Xenium_SpatialData_Breast",
            file_path=file_path,
            dimensions_level_0=(int(w), int(h)),
            mpp=self.mpp,
            staining="Xenium_H&E_Aligned",
            levels_count=1,
            levels_info=[WSILevelMetadata(level=0, dimensions=(int(w), int(h)), downsample=1.0)],
        )

    def load_region(self, container: WSIContainer, x: int, y: int, width: int, height: int, level: int = 0) -> WSIRegionContainer:
        """
        Загружает RGB-патч изображения, используя пространственный запрос SpatialData.
        """
        # Делаем срез (crop) изображения по координатам bounding box (в пиксельной/глобальной системе)
        cropped_sdata = bounding_box_query(
            self.sdata,
            axes=('x', 'y'),
            min_coordinate=[x, y],
            max_coordinate=[x + width, y + height],
            target_coordinate_system=self.coordinate_system
        )
        
        # Извлекаем numpy-массив из полученного под-объекта
        image_tensor = cropped_sdata.images[self.image_key].values
        
        # Приводим размерности к виду (C, H, W) -> стандарт для PyTorch
        if image_tensor.ndim == 4:  # Если есть лишняя ось (например, dummy-ось времени или z-стека)
            image_tensor = image_tensor.squeeze(0)
            
        # Если каналы оказались в конце (H, W, C), переносим их вперед
        if image_tensor.shape[-1] == 3 or image_tensor.shape[-1] == 4:
            image_tensor = np.moveaxis(image_tensor, -1, 0)
            
        # Оставляем только первые 3 канала (RGB), если есть альфа-канал
        image_tensor = image_tensor[:3, ...]
        
        # Нормализация интенсивности пикселей в [0, 1]
        image_tensor = image_tensor.astype(np.float32)
        if image_tensor.max() > 1.0:
            p2, p98 = np.percentile(image_tensor, (2, 98))
            if p98 - p2 > 0:
                image_tensor = np.clip((image_tensor - p2) / (p98 - p2), 0, 1)
            else:
                image_tensor = image_tensor / image_tensor.max()

        return WSIRegionContainer(
            slide_id=container.slide_id,
            image_tensor=torch.from_numpy(image_tensor),
            x_wsi=int(x), y_wsi=int(y),
            width_wsi=int(width), height_wsi=int(height),
            extracted_level=level,
            mpp=self.mpp,
        )

    def get_cells_in_region(self, x: int, y: int, width: int, height: int) -> List[CellFeaturePassport]:
        """
        Вырезает полигоны клеток, попавшие в bounding box, без ручных циклов фильтрации.
        """
        # Запрашиваем геометрию только для нужного региона
        cropped_sdata = bounding_box_query(
            self.sdata,
            axes=('x', 'y'),
            min_coordinate=[x, y],
            max_coordinate=[x + width, y + height],
            target_coordinate_system=self.coordinate_system
        )
        
        region_shapes = cropped_sdata.shapes[self.shapes_key]
        passports = []
        
        for idx, row in region_shapes.iterrows():
            # Извлекаем Shapely-полигон напрямую
            geom = row["geometry"]
            if geom is None or geom.is_empty:
                continue
                
            # Получаем список координат вершин полигона (уже в пикселях/глобальной системе!)
            poly_points = list(geom.exterior.coords)
            
            # Вычисляем центроид полигона
            centroid_point = geom.centroid
            cx, cy = centroid_point.x, centroid_point.y
            
            # Берем индекс строки. В SpatialData индекс Shapes соответствует cell_id из таблицы экспрессии
            cell_id = str(idx)
            
            m = self.patch_margin
            poly = PolygonModel(points=poly_points)
            
            segmentation = SegmentationCellInstance(
                polygon=poly,
                bbox=(int(cx - m), int(cy - m), int(m * 2), int(m * 2)),
                score=1.0,
                centroid=(cx, cy),
            )
            
            passports.append(CellFeaturePassport(
                slide_id="Xenium_SpatialData_Breast",
                cell_id=cell_id,
                segmentation_data=segmentation,
            ))
            
        return passports
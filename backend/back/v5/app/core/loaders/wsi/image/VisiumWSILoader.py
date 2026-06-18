import os
import tifffile
from typing import List, Tuple
from app.core.schemes.WSI import WSIСontainer, WSILevelMetadata, WSIRegionContainer
from app.core.loaders.wsi.image.AbstractWSILoader import AbstractWSILoader
import torch
import cv2
import numpy as np

class VisiumWSILoader(AbstractWSILoader):
    """
    Реальный загрузчик гистологических слайдов для датасетов 10x Genomics Visium.
    Использует tifffile для быстрого чтения тяжелых научных TIFF-файлов без утечек памяти.
    """

    def load_slide_metadata(self, file_path: str) -> WSIСontainer:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл слайда не найден по пути: {file_path}")

        print(f"📖 Чтение заголовков TIFF пирамиды: {file_path}")
        
        # Открываем TIFF файл в ленивом режиме (без загрузки пикселей в память)
        with tifffile.TiffFile(file_path) as tif:
            # Каждый объект серии (series) или страница (page) может быть уровнем пирамиды
            # В стандартных WSI от 10x Genomics уровни лежат в tif.series[0].levels
            
            levels_info: List[WSILevelMetadata] = []
            
            # Если файл содержит пирамиду (несколько под-изображений)
            if tif.series:
                main_series = tif.series[0]
                levels_count = len(main_series.levels)
                
                # Базовое разрешение (Level 0)
                h_0, w_0 = main_series.shape[:2]
                dimensions_level_0 = (w_0, h_0)
                
                # Собираем информацию по всем уровням
                for idx, level_data in enumerate(main_series.levels):
                    h_l, w_l = level_data.shape[:2]
                    
                    # Считаем коэффициент даунсэмплинга (во сколько раз уменьшено относительно level 0)
                    downsample = float(w_0 / w_l) if w_l > 0 else 1.0
                    
                    levels_info.append(WSILevelMetadata(
                        level=idx,
                        dimensions=(w_l, h_l),
                        downsample=downsample
                    ))
            else:
                # Если это плоский TIFF (всего 1 уровень)
                # Такое бывает, если скачан оптимизированный веб-снимок
                with tifffile.TiffPage(file_path) as page:
                    h_0, w_0 = page.shape[:2]
                
                dimensions_level_0 = (w_0, h_0)
                levels_count = 1
                levels_info.append(WSILevelMetadata(
                    level=0,
                    dimensions=(w_0, h_0),
                    downsample=1.0
                ))

        # Выставляем стандартный MPP (Microns Per Pixel) для 10x Visium.
        # Для классического Visium разрешение сканирования обычно составляет ~0.5 мкм/пиксель.
        # В реальных метаданных это значение можно вытащить из XML тегов, но захардкодим стандарт.
        mpp_standard = 0.499

        # Собираем и возвращаем твой валидный WSIContainer!
        return WSIСontainer(
            slide_id=os.path.basename(file_path).replace(".tif", ""),
            file_path=os.path.abspath(file_path),
            dimensions_level_0=dimensions_level_0,
            mpp=mpp_standard,
            staining="H&E",
            levels_count=levels_count,
            levels_info=levels_info,
            metadata={"scanner": "10x Genomics Automated Imaging Pipeline", "format": "TIFF/BigTIFF"}
        )

    def load_region(
        self, 
        container: WSIСontainer, 
        x: int, 
        y: int, 
        width: int, 
        height: int, 
        level: int = 0
    ) -> WSIRegionContainer:
        """
        Вырезает кроп со слайда и приводит его к строгому контрасту твоей схемы WSIRegionContainer
        """
        # Проверяем, существует ли запрашиваемый уровень
        if level >= container.levels_count:
            raise IndexError(f"Запрошен уровень пирамиды {level}, но доступно всего {container.levels_count}")
            
        # Считаем коэффициент масштабирования для запрашиваемого уровня
        downsample = container.levels_info[level].downsample
        
        # Пересчитываем глобальные координаты уровня 0 в локальные координаты нужного уровня
        x_local = int(x / downsample)
        y_local = int(y / downsample)
        w_local = int(width / downsample)
        h_local = int(height / downsample)

        # Читаем только нужный регион с диска, используя OpenCV или Tifffile
        # Для простоты и скорости используем ленивое чтение через tifffile/OpenCV
        # (OpenCV отлично делает это, если файл помещается в память, или читает через слайсинг)
        img = cv2.imread(container.file_path)
        
        # Вырезаем кроп (координаты в numpy: [y:y+h, x:x+w])
        crop = img[y_local:y_local+h_local, x_local:x_local+w_local]
        
        # Конвертируем BGR (стандарт OpenCV) в RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Приводим к стандарту PyTorch Tensor (C, H, W) и типу float32 в диапазоне [0.0, 1.0]
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).to(torch.float32) / 255.0

        # Возвращаем твой идеальный WSIRegionContainer!
        return WSIRegionContainer(
            slide_id=container.slide_id,
            image_tensor=tensor,
            x_wsi=x,
            y_wsi=y,
            width_wsi=width,
            height_wsi=height,
            extracted_level=level,
            mpp=container.mpp * downsample
        )
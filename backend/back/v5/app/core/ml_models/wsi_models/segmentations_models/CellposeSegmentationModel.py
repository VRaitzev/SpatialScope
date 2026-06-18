import numpy as np
import torch
from typing import List, Dict, Tuple
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Импортируем твои базовые абстракции и Pydantic-схемы
# (Пути импортов адаптируй под реальную структуру папок твоего проекта app/)
from app.core.ml_models.wsi_models.segmentations_models import AbstractSegmentationModel
from app.core.schemes.WSI import WSIRegionContainer
from app.core.schemes.Segmentation import (
    SegmentationContainer, 
    SegmentationCellInstance, 
    PolygonModel, 
    SegmentationFeatureValue,
    SegmentationFeatureMeta,
    SegmentationModelManifest
)

class StarDistSegmentationModel(AbstractSegmentationModel):
    """
    [MODEL LAYER] Реализация модели сегментации ядер клеток на базе StarDist 2D.
    Специализированный бейзлайн для H&E гистологии рака молочной железы.
    """

    def __init__(self):
        self._model: StarDist2D | None = None
        
        # Строгое декларирование фичей, которые мы будем отдавать в граф
        self._manifest = SegmentationModelManifest(
            model_name="stardist_2d_versatile_he",
            model_version="0.7.3_pretrained",
            model_description=(
                "Сегментация ядер клеток на гистологических срезах H&E с использованием "
                "архитектуры звездно-выпуклых полигонов. Оптимальна для плотных опухолевых кластеров."
            ),
            supported_features=[
                SegmentationFeatureMeta(
                    feature_name="nuclear_area", 
                    description="Площадь ядра клетки в пикселях", 
                    unit="px²"
                ),
                SegmentationFeatureMeta(
                    feature_name="nuclear_perimeter", 
                    description="Периметр ядра клетки, рассчитанный через геометрию полигона", 
                    unit="px"
                )
            ]
        )

    @property
    def manifest(self) -> SegmentationModelManifest:
        """Возвращает манифест модели для регистрации в глобальном реестре пайплайна"""
        return self._manifest

    def load_weights(self, path: str = None) -> None:
        """
        Загрузка предобученных весов '2D_versatile_he'.
        Аргумент 'path' опционален, так как веса скачиваются автоматически библиотекой StarDist.
        """
        print("🚀 [MODEL] Инициализация весов StarDist (2D_versatile_he)...")
        
        # StarDist использует TensorFlow под капотом. 
        # Он автоматически займет доступную видеопамять твоей GTX 1650 Super.
        self._model = StarDist2D.from_pretrained('2D_versatile_he')
        
        print("✅ [MODEL] Модель StarDist успешно загружена и готова к инференсу.")

    def save_weights(self, path: str) -> None:
        """Для встроенных предобученных весов сохранение не требуется"""
        if self._model is not None:
            self._model.keras_model.save_weights(path)
            print(f"💾 [MODEL] Веса модели сохранены в: {path}")

    def predict(self, patch: WSIRegionContainer) -> SegmentationContainer:
        """
        Основной forward-pass инференса сегментации ядер.
        Берет патч из WSI-лоадера, переводит в NumPy, нормализует и извлекает полигоны.
        """
        if self._model is None:
            raise RuntimeError(
                "Попытка вызвать predict() до инициализации модели! "
                "Сначала необходимо вызвать метод load_weights()."
            )

        # === ЭТАП 1: ТРАНСФОРМАЦИЯ И НОРМАЛИЗАЦИЯ ДАННЫХ ===
        # 1. Забираем тензор из WSIRegionContainer. Он имеет форму (C, H, W) и тип float32.
        tensor = patch.image_tensor.cpu()
        
        # 2. Разворачиваем оси: (C, H, W) -> (H, W, C), как требует компьютерное зрение
        img_numpy = tensor.permute(1, 2, 0).numpy()
        
        # 3. Возвращаем пиксели из диапазона [0.0, 1.0] в стандартный диапазон [0, 255]
        img_numpy = (img_numpy * 255.0).astype(np.uint8)

        # 4. Профессиональная нормализация StarDist (из библиотеки csbdeep).
        # Она отсекает 1% самых темных и 0.2% самых светлых пикселей (квантили),
        # выравнивая контрастность фиолетовых ядер, если слайд перекрашен или смазан.
        img_normalized = normalize(img_numpy, 1, 99.8, axis=(0, 1))

        # === ЭТАП 2: ИНФЕРЕНС СЕТИ ===
        # labels: Матрица масок (размер HxW), где фон — 0, а ядра — целые числа 1, 2... N
        # details: Словарь с геометрией полигонов:
        #   details['points'] -> центры [Y, X]
        #   details['coord']  -> координаты лучей границы
        #   details['prob']   -> уверенность сети в каждом ядре [0.0, 1.0]
        labels, details = self._model.predict_instances(img_normalized)

        instances: List[SegmentationCellInstance] = []
        num_cells = len(details['points'])

        # === ЭТАП 3: ЭКСТРАКЦИЯ ГЕОМЕТРИИ И ВАЛИДАЦИЯ СХЕМ ===
        for i in range(num_cells):
            # 1. Извлекаем предсказанный StarDist центр ядра.
            # Внимание: StarDist выдает координаты в матричном формате [Y, X].
            # Мы переворачиваем их в привычный геометрический формат [X, Y].
            coord_y, coord_x = details['points'][i]
            centroid_coord = (float(coord_x), float(coord_y))

            # 2. Извлекаем лучи полигона границы.
            # details['coord'][i] имеет размерность (2, Количество_лучей), где строка 0 - это Y, строка 1 - это X
            poly_y = details['coord'][i][0]
            poly_x = details['coord'][i][1]
            
            # Собираем точки в список кортежей (x, y) для PolygonModel
            polygon_points = [(float(x), float(y)) for x, y in zip(poly_x, poly_y)]

            # Проверяем геометрическую валидность (для полигона нужно минимум 3 точки)
            if len(polygon_points) < 3:
                continue

            # Создаем объект PolygonModel из твоего контракта
            polygon_model = PolygonModel(points=polygon_points)

            # 3. Вычисляем Bounding Box (минимальную рамку, описывающую ядро)
            ymin, xmin = float(np.min(poly_y)), float(np.min(poly_x))
            ymax, xmax = float(np.max(poly_y)), float(np.max(poly_x))
            bbox = (int(ymin), int(xmin), int(ymax), int(xmax))

            # 4. Рассчитываем биологические фичи ядра через Shapely встроенный метод
            # Мы используем твой метод .as_shapely(), чтобы получить честную геометрию
            shapely_poly = polygon_model.as_shapely()
            
            features: Dict[str, SegmentationFeatureValue] = {
                "nuclear_area": SegmentationFeatureValue(
                    feature_name="nuclear_area", 
                    value=float(shapely_poly.area)
                ),
                "nuclear_perimeter": SegmentationFeatureValue(
                    feature_name="nuclear_perimeter", 
                    value=float(shapely_poly.length)
                )
            }

            # 5. Сборка валидного инстанса клетки.
            # Благодаря твоему @model_validator(mode='after'), если centroid_coord 
            # вдруг смажется, Pydantic сам пересчитает идеальный центр масс полигона!
            cell_instance = SegmentationCellInstance(
                polygon=polygon_model,
                bbox=bbox,
                score=float(details['prob'][i]),
                centroid=centroid_coord,
                features=features
            )
            
            instances.append(cell_instance)

        # === ЭТАП 4: УПАКОВКА В ИТОГОВЫЙ КОНТЕЙНЕР ===
        return SegmentationContainer(
            instances=instances,
            image_shape=img_numpy.shape[:2],
            model_name=self._manifest.model_name,
            version=self._manifest.model_version
        )
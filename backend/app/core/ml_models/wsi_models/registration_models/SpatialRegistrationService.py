"""
Сервис пространственного совмещения мультимодальных биомедицинских данных.
Реализует кэширование результатов нелинейной регистрации на диске с автоматическим 
отсечением пустого фона слайда и фильтрацией артефактов сканирования.
"""

import os
import json
import numpy as np
import pandas as pd
import ants
import tifffile
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from app.core.schemes.Segmentation import SegmentationContainer, SegmentationCellInstance


class SpatialRegistrationService:
    def __init__(self, 
                 dapi_path: str, 
                 he_path: str, 
                 cache_dir: str, 
                 target_max_dim: int = 4000, 
                 mpp_xenium: float = 0.2125):
        """
        Инициализация сервиса. Проверяет наличие кэша. 
        Если кэш пуст — запускает тяжелую регистрацию и сохраняет её результаты.
        """
        self.mpp_xenium = mpp_xenium
        self.cache_dir = cache_dir
        
        # Создаем директорию для кэша, если её ещё нет
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Префикс для файлов, которые сгенерирует ANTsPy (матрицы и поля деформаций)
        self.out_prefix = os.path.join(self.cache_dir, "transform_")
        self.meta_path = os.path.join(self.cache_dir, "registration_meta.json")
        
        if self._is_cached():
            print("📦 [RegistrationService] Найден готовый кэш регистрации. Мгновенная загрузка...")
            self.reg = self._load_from_cache()
        else:
            print("🔧 [RegistrationService] Кэш не найден. Запуск тяжелой нелинейной SyN регистрации...")
            self.reg = self._register_and_cache(dapi_path, he_path, target_max_dim)
            print("✅ [RegistrationService] Глобальные поля трансформации успешно вычислены и сохранены в кэш.")

    def _is_cached(self) -> bool:
        """Проверяет, существует ли валидный файл метаданных и сами файлы трансформаций."""
        if not os.path.exists(self.meta_path):
            return False
            
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
            # Проверяем физическое существование всех файлов трансформаций
            for file_path in meta.get("fwdtransforms", []) + meta.get("invtransforms", []):
                if not os.path.exists(file_path):
                    return False
            return True
        except Exception:
            return False

    def _load_from_cache(self) -> dict:
        """Считывает параметры деформации и смещения кропа из сохраненного json-файла конфигурации."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        return {
            "fwdtransforms": meta["fwdtransforms"],
            "invtransforms": meta["invtransforms"],
            "dapi_downsample": meta["dapi_downsample"],
            "he_downsample": meta["he_downsample"],
            "dapi_crop_offset": meta["dapi_crop_offset"],
            "he_crop_offset": meta["he_crop_offset"]
        }

    def _compute_foreground_bbox(self, arr: np.ndarray, is_he: bool, margin_frac: float = 0.03):
        """
        Находит bbox реальной ткани на уровне пирамиды, разделяя логику 
        для светлого поля (H&E) и темного поля (DAPI флуоресценция).
        """
        lum = arr.mean(axis=-1) if arr.ndim == 3 else arr
        lum = lum.astype(np.float32)
        
        if is_he:
            # Для H&E (после очистки рамок): ткань гарантированно ТЕМНЕЕ чисто белого фона
            max_val = lum.max()
            try:
                thresh = threshold_otsu(lum)
                mask = lum < (max_val - (max_val - thresh) * 0.6)
            except ValueError:
                mask = lum < (max_val * 0.9)
        else:
            # Для DAPI: ткань гарантированно ЯРЧЕ черного фона
            try:
                thresh = threshold_otsu(lum)
                mask = lum > (thresh * 0.4)
            except ValueError:
                mask = lum > (lum.max() * 0.1)
        
        # Очищаем маску от мелких изолированных пикселей-шумов
        mask = ndi.binary_opening(mask, iterations=2)
        mask = ndi.binary_closing(mask, iterations=2)
        
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return 0, 0, arr.shape[1], arr.shape[0]
            
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        
        # Добавляем небольшие безопасные отступы вокруг ткани (margin)
        mx = int((x1 - x0) * margin_frac)
        my = int((y1 - y0) * margin_frac)
        
        x0, y0 = max(0, x0 - mx), max(0, y0 - my)
        x1, y1 = min(arr.shape[1], x1 + mx), min(arr.shape[0], y1 + my)
        
        return x0, y0, x1, y1

    def _register_and_cache(self, dapi_path: str, he_path: str, target_max_dim: int) -> dict:
        """Выполняет SyN регистрацию кропнутых изображений тканей и записывает файлы на диск."""
        dapi_img, dapi_ds, dapi_offset = self._load_downsampled_gray(dapi_path, target_max_dim, is_he=False)
        he_img, he_ds, he_offset = self._load_downsampled_gray(he_path, target_max_dim, is_he=True)

        fixed = ants.from_numpy(he_img.astype(np.float32))
        moving = ants.from_numpy(dapi_img.astype(np.float32))

        # Передаем outprefix, чтобы ANTsPy сохранил файлы прямо в папку кэша
        result = ants.registration(
            fixed=fixed, moving=moving,
            type_of_transform="SyNRA",
            outprefix=self.out_prefix
        )

        reg_data = {
            "fwdtransforms": result["fwdtransforms"],
            "invtransforms": result["invtransforms"],
            "dapi_downsample": dapi_ds,
            "he_downsample": he_ds,
            "dapi_crop_offset": list(dapi_offset),
            "he_crop_offset": list(he_offset)
        }

        # Сохраняем метаданные в JSON для последующих быстрых запусков
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(reg_data, f, ensure_ascii=False, indent=4)

        return reg_data

    def _load_downsampled_gray(self, path: str, target_max_dim: int, is_he: bool):
        """Загружает уровень пирамиды, убирает артефакты сканера, кропает ткань и нормализует."""
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            full_dims = max(d for d in series.shape if d > 4)
            levels = series.levels if hasattr(series, "levels") else [series]

            best_level = 0
            for i, lvl in enumerate(levels):
                dims = max(d for d in lvl.shape if d > 4)
                best_level = i
                if dims <= target_max_dim:
                    break

            arr = levels[best_level].asarray()
            downsample = full_dims / max(d for d in arr.shape if d > 4)

        arr = np.squeeze(arr)

        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.moveaxis(arr, 0, -1)

        # 🔥 КРИТИЧЕСКИЙ ФИКС: Заменяем темные артефакты рамки сканирования H&E на белый цвет (255)
        # Это предотвращает логарифмический взрыв в rgb2hed и полностью возвращает контраст ткани.
        if is_he:
            if arr.ndim == 3:
                is_black_border = np.all(arr < 35, axis=-1)
                arr[is_black_border] = 255
            else:
                is_black_border = arr < 35
                arr[is_black_border] = 255

        # Вычисляем bounding box ткани с учетом типа изображения
        x0, y0, x1, y1 = self._compute_foreground_bbox(arr, is_he=is_he)
        arr = arr[y0:y1, x0:x1]
        
        # Переводим смещение кропа в полноразмерные пиксели исходного WSI-слайда
        crop_offset_full = (x0 * downsample, y0 * downsample)

        if is_he:
            arr3 = arr[..., :3].astype(np.float32) / (arr.max() + 1e-6)
            gray = -rgb2hed(arr3)[..., 0]
        else:
            if arr.ndim == 3:
                if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                    arr = arr[0]
                else:
                    arr = arr[..., 0]
            gray = arr.astype(np.float32)

        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
        return gray, downsample, crop_offset_full

    def warp_xenium_pixels_to_he_pixels(self, points_xenium_px: np.ndarray) -> np.ndarray:
        """
        Трансформирует координаты из пространства полноразмерных пикселей Xenium в пиксели H&E.
        Учитывает смещения кропа ткани с обеих сторон.
        """
        pts = np.asarray(points_xenium_px, dtype=np.float64)
        is_single = False

        # 1. Проверяем размерность входных данных
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
            is_single = True
        elif pts.ndim == 2 and pts.shape[0] == 1:
            is_single = True

        # Сдвигаем оригинальные точки Xenium внутрь кропнутой области DAPI
        dx, dy = self.reg["dapi_crop_offset"]
        pts_cropped = pts - np.array([dx, dy])

        # Переводим точки в downsample-координаты кропнутой области
        pts_ds = pts_cropped / self.reg["dapi_downsample"]

        # Хак для обхода бага ITK CSV-парсера на Windows для одиночной строки
        if is_single:
            pts_ds = np.vstack([pts_ds, pts_ds])

        # Создаем DataFrame для ANTsPy
        pts_df = pd.DataFrame(pts_ds, columns=['x', 'y'])

        # Вызываем деформацию точек ANTsPy
        warped_df = ants.apply_transforms_to_points(
            dim=2,
            points=pts_df,
            transformlist=self.reg['invtransforms']  # <--- Поменяли здесь
        )

        result_ds = warped_df[['x', 'y']].to_numpy()

        # Возвращаем деформированные точки обратно в полный масштаб и добавляем смещение H&E
        hx, hy = self.reg["he_crop_offset"]
        result_full = result_ds * self.reg["he_downsample"] + np.array([hx, hy])

        if is_single:
            return result_full[0:1]
        
        return result_full

    def _warp_points(self, points_um: np.ndarray) -> np.ndarray:
        """Внутренний математический метод трансформации массива точек из микрометров (um)."""
        if len(points_um) == 0:
            return points_um
        points_px = points_um / self.mpp_xenium
        return self.warp_xenium_pixels_to_he_pixels(points_px)

    def warp_container(self, native_container: SegmentationContainer) -> SegmentationContainer:
        """Интерфейсный метод деформации геометрии Pydantic-контейнера клеток."""
        warped_instances = []
        
        for cell in native_container.instances:
            native_polygon = np.array(cell.polygon)
            native_centroid = np.array([cell.centroid])
            
            warped_polygon = self._warp_points(native_polygon)
            warped_centroid = self._warp_points(native_centroid)[0]
            
            min_x, min_y = np.min(warped_polygon, axis=0)
            max_x, max_y = np.max(warped_polygon, axis=0)
            warped_bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            
            warped_cell = SegmentationCellInstance(
                polygon=warped_polygon.tolist(),
                bbox=warped_bbox,
                score=cell.score,
                centroid=warped_centroid.tolist(),
                features=cell.features 
            )
            warped_instances.append(warped_cell)
            
        return SegmentationContainer(
            instances=warped_instances,
            image_shape=native_container.image_shape,
            model_name=f"{native_container.model_name}_ANTsPy_Warped",
            version=native_container.version
        )
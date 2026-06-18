import pandas as pd
import tifffile
import torch
import numpy as np
import os
from typing import List
from app.core.loaders.wsi.image.AbstractWSILoader import AbstractWSILoader
from app.core.schemes.WSI import WSIContainer, WSIRegionContainer, WSILevelMetadata
from app.core.schemes.Cells import CellFeaturePassport
from app.core.schemes.Segmentation import SegmentationCellInstance, PolygonModel


def _normalize_cell_id(raw_id) -> str:
    """
    Приводит cell_id к единому строковому виду независимо от того,
    как он был прочитан из parquet: bytes, обычная str, либо
    str-репрезентация bytes ("b'aaaaaaaa-1'").
    Применять ОБЯЗАТЕЛЬНО к обоим датафреймам ДО merge/groupby —
    иначе при несовпадении форматов .map() вернёт NaN для всех строк.
    """
    if isinstance(raw_id, (bytes, bytearray)):
        cid = raw_id.decode("utf-8")
    else:
        cid = str(raw_id).strip()

    if (cid.startswith("b'") and cid.endswith("'")) or (cid.startswith('b"') and cid.endswith('"')):
        cid = cid[2:-1]

    return cid


class XeniumWSILoader(AbstractWSILoader):
    def __init__(self, cells_path: str, patch_margin: int = 15):
        self.patch_margin = patch_margin
        self.cells_df = pd.read_parquet(cells_path)

        base_path = os.path.dirname(cells_path)
        self.boundaries_path = os.path.join(base_path, "cell_boundaries.parquet")
        self.boundaries_df = pd.read_parquet(self.boundaries_path)

        self.cells_df["cell_id"] = self.cells_df["cell_id"].apply(_normalize_cell_id)
        self.boundaries_df["cell_id"] = self.boundaries_df["cell_id"].apply(_normalize_cell_id)

        grouped = self.boundaries_df.groupby("cell_id")[["vertex_x", "vertex_y"]].apply(
            lambda x: list(zip(x["vertex_x"], x["vertex_y"]))
        )
        self.cells_df["polygon_points"] = self.cells_df["cell_id"].map(grouped)
        self.mpp = 0.2125  # Стандарт для Xenium
        
    # def apply_registration_to_dataframe(self, reg_service=None):
    #     """
    #     Интеграция с сервисом нелинейной регистрации.
    #     Если reg_service передан — считаем координаты через ANTsPy поля деформаций.
    #     Если None — падаем в дефолтный MPP-скейлинг.
    #     """
    #     if reg_service is not None:
    #         print("🔄 [Loader] Запуск нелинейного пересчета H&E координат через ANTsPy...")
            
    #         # Переводим микрометры центроидов в нативные пиксели Xenium
    #         cx_px = self.cells_df["x_centroid"].values / self.mpp
    #         cy_px = self.cells_df["y_centroid"].values / self.mpp
    #         pts_px = np.stack([cx_px, cy_px], axis=1)
            
    #         # Магический прогон массива точек через поля деформаций
    #         pts_he = reg_service.warp_xenium_pixels_to_he_pixels(pts_px)
            
    #         self.cells_df["x_he_px"] = pts_he[:, 0]
    #         self.cells_df["y_he_px"] = pts_he[:, 1]
    #         print("✅ Координаты нелинейной регистрации успешно интегрированы.")
    #     else:
    #         print("🔄 [Loader] Расчет H&E координат напрямую по MPP (без деформации)...")
    #         self.cells_df["x_he_px"] = self.cells_df["x_centroid"].values / self.mpp
    #         self.cells_df["y_he_px"] = self.cells_df["y_centroid"].values / self.mpp
    #     print("✅ Координаты MPP интегрированы.")


    def apply_registration_to_dataframe(self, reg_service=None):
        """
        Для выровненного датасета нам больше не нужно деформировать координаты через ANTsPy.
        Мы просто переводим микрометры в пиксели по константе MPP.
        """
        print("🔄 [Loader] Расчет H&E координат напрямую (без деформации)...")
        
        # Просто переводим микрометры центра клеток в пиксели
        self.cells_df["x_he_px"] = self.cells_df["x_centroid"].values / self.mpp
        self.cells_df["y_he_px"] = self.cells_df["y_centroid"].values / self.mpp
        
        print("✅ Координаты H&E успешно интегрированы.")

    def load_slide_metadata(self, file_path: str) -> WSIContainer:
        """
        Считывает размеры из реального H&E файла, корректно обрабатывая 
        различные варианты упорядочивания осей (Channels-first / Channels-last).
        """
        with tifffile.TiffFile(file_path) as tif:
            shape = tif.series[0].shape
            
            # Если изображение трехканальное (RGB)
            if len(shape) == 3:
                if shape[0] in (3, 4):  # Формат (Channels, Height, Width)
                    h, w = shape[1], shape[2]
                elif shape[-1] in (3, 4):  # Формат (Height, Width, Channels)
                    h, w = shape[0], shape[1]
                else:
                    h, w = shape[0], shape[1]
            else:
                # Универсальный фолбэк для сложных пирамид: выбираем две самые большие оси
                spatial_dims = [d for d in shape if d > 4]
                h, w = spatial_dims[0], spatial_dims[1]

        return WSIContainer(
            slide_id="Xenium_Paired_Breast",
            file_path=file_path,
            dimensions_level_0=(w, h),  # Контракт требует (Width, Height)
            mpp=self.mpp,
            staining="H&E_Registered",
            levels_count=1,
            levels_info=[WSILevelMetadata(level=0, dimensions=(w, h), downsample=1.0)],
        )

    def load_region(self, container: WSIContainer, x: int, y: int, width: int, height: int, level: int = 0) -> WSIRegionContainer:
        with tifffile.TiffFile(container.file_path) as tif:
            series = tif.series[0]
            # Защита от многоканальности fixed-файла, если она есть
            crop = series.asarray(level=level)

        if crop.ndim > 3:
            crop = crop.squeeze()
        if crop.ndim == 3 and crop.shape[0] <= 4:
            crop = np.moveaxis(crop, 0, -1)

        crop_region = crop[y:y + height, x:x + width]
        if crop_region.ndim == 2:
            crop_rgb = np.stack([crop_region] * 3, axis=-1)
        elif crop_region.shape[-1] == 3:
            crop_rgb = crop_region
        else:
            crop_rgb = crop_region[..., :3]

        crop_rgb = crop_rgb.astype(np.float32)
        if crop_rgb.max() > 1.0:
            crop_rgb = crop_rgb / 255.0  # Обычный скейлинг для H&E

        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).to(torch.float32)
        return WSIRegionContainer(
            slide_id=container.slide_id, image_tensor=tensor,
            x_wsi=int(x), y_wsi=int(y), width_wsi=int(width), height_wsi=int(height),
            extracted_level=level, mpp=self.mpp,
        )

    def get_cells_in_region(self, x: int, y: int, width: int, height: int, use_he: bool = False) -> List[CellFeaturePassport]:
        print(self.boundaries_df[["vertex_x","vertex_y"]].describe())
        print(self.cells_df[["x_centroid","y_centroid"]].describe())
        print(self.boundaries_df.columns)
        # Если use_he=True, фильтруем по вычисленным H&E координатам
        if use_he and "x_he_px" in self.cells_df.columns:
            mask = (
                (self.cells_df["x_he_px"] >= x) & (self.cells_df["x_he_px"] < x + width) &
                (self.cells_df["y_he_px"] >= y) & (self.cells_df["y_he_px"] < y + height)
            )
        else:
            x_um, y_um = x * self.mpp, y * self.mpp
            w_um, h_um = width * self.mpp, height * self.mpp
            mask = (
                (self.cells_df["x_centroid"] >= x_um) & (self.cells_df["x_centroid"] < x_um + w_um) &
                (self.cells_df["y_centroid"] >= y_um) & (self.cells_df["y_centroid"] < y_um + h_um)
            )
        
        subset = self.cells_df[mask]
        passports = []
        for _, row in subset.iterrows():
            m = self.patch_margin
            cx = float(row["x_centroid"]) / self.mpp
            cy = float(row["y_centroid"]) / self.mpp

            points = row["polygon_points"]
            poly_points = [(px / self.mpp, py / self.mpp) for px, py in points] if isinstance(points, list) else []

            poly = PolygonModel(points=poly_points)
            segmentation = SegmentationCellInstance(
                polygon=poly,
                bbox=(int(cx - m), int(cy - m), int(m * 2), int(m * 2)),
                score=1.0, centroid=(cx, cy),
            )
            passports.append(CellFeaturePassport(
                slide_id="Xenium_Paired_Breast", cell_id=row["cell_id"], segmentation_data=segmentation,
            ))
        return passports
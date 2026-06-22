import os
from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
import tifffile
import torch
import zarr
from PIL import Image

from app.core.loaders.wsi.image.AbstractWSILoader import AbstractWSILoader
from app.core.schemes.Cells import CellFeaturePassport
from app.core.schemes.Segmentation import PolygonModel, SegmentationCellInstance
from app.core.schemes.WSI import WSIContainer, WSILevelMetadata, WSIRegionContainer


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

    if (cid.startswith("b'") and cid.endswith("'")) or (
        cid.startswith('b"') and cid.endswith('"')
    ):
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
        self.boundaries_df["cell_id"] = self.boundaries_df["cell_id"].apply(
            _normalize_cell_id
        )

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

    def _read_level_region(
        self,
        file_path: str,
        level: int = 0,
        *,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> Tuple[np.ndarray, str]:
        with tifffile.TiffFile(file_path) as tif:
            series = tif.series[0]
            levels = (
                series.levels
                if hasattr(series, "levels") and series.levels
                else [series]
            )
            selected_level = min(level, len(levels) - 1)
            level_series = levels[selected_level]
            axes = getattr(level_series, "axes", getattr(series, "axes", "YX"))
            shape = tuple(level_series.shape)
            y_size = int(shape[axes.index("Y")]) if "Y" in axes else int(shape[-2])
            x_size = int(shape[axes.index("X")]) if "X" in axes else int(shape[-1])

            x = max(0, min(int(x), max(x_size - 1, 0)))
            y = max(0, min(int(y), max(y_size - 1, 0)))
            width = max(1, min(int(width or x_size), x_size - x))
            height = max(1, min(int(height or y_size), y_size - y))

            store = series.aszarr()
            zarr_group = zarr.open(store, mode="r")
            arr = zarr_group[str(selected_level)]

            slices = []
            for axis, axis_size in zip(axes, arr.shape):
                if axis == "Y":
                    slices.append(slice(y, y + height))
                elif axis == "X":
                    slices.append(slice(x, x + width))
                else:
                    slices.append(slice(None))
            region = np.asarray(arr[tuple(slices)])

        return region, axes

    def _collapse_to_image(self, arr: np.ndarray, axes: str) -> np.ndarray:
        image = np.asarray(arr)
        image_axes = axes

        while image.ndim > len(image_axes) and image.ndim > 0:
            image = image[0]

        if "Z" in image_axes and image.ndim >= 3:
            z_idx = image_axes.index("Z")
            image = image.max(axis=z_idx)
            image_axes = image_axes.replace("Z", "")

        if "C" in image_axes and image.ndim >= 3:
            c_idx = image_axes.index("C")
            if c_idx != image.ndim - 1:
                image = np.moveaxis(image, c_idx, -1)
                image_axes = image_axes.replace("C", "") + "C"
        elif "S" in image_axes and image.ndim >= 3:
            s_idx = image_axes.index("S")
            if s_idx != image.ndim - 1:
                image = np.moveaxis(image, s_idx, -1)
                image_axes = image_axes.replace("S", "") + "S"
        elif (
            image.ndim == 3
            and image.shape[0] in (3, 4)
            and image.shape[-1] not in (3, 4)
        ):
            image = np.moveaxis(image, 0, -1)

        while image.ndim > 3:
            image = image[0]

        return image

    def _ensure_rgb_uint8(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            max_val = float(arr.max()) if arr.size else 1.0
            if max_val > 0:
                scale = (
                    255.0 / max_val if max_val <= 1.0 else 255.0 / max(max_val, 255.0)
                )
                arr = np.clip(arr * scale, 0, 255)
            arr = arr.astype(np.uint8)
        return arr

    def _pick_level_for_downsample(
        self, container: WSIContainer, desired_downsample: float
    ) -> int:
        if not container.levels_info:
            return 0
        return min(
            range(len(container.levels_info)),
            key=lambda idx: abs(
                float(container.levels_info[idx].downsample) - float(desired_downsample)
            ),
        )

    def load_slide_metadata(self, file_path: str) -> WSIContainer:
        """
        Считывает размеры из реального H&E файла и старается корректно определить
        пространственные оси даже для сложных OME-TIFF представлений.
        """
        with tifffile.TiffFile(file_path) as tif:
            series = tif.series[0]
            levels_info = []

            if hasattr(series, "levels") and series.levels:
                base_w = base_h = None
                for idx, level in enumerate(series.levels):
                    axes = getattr(level, "axes", getattr(series, "axes", ""))
                    shape = tuple(level.shape)
                    axis_map = {
                        axis: shape[pos]
                        for pos, axis in enumerate(axes)
                        if pos < len(shape)
                    }
                    width = int(axis_map.get("X", shape[-1]))
                    height = int(
                        axis_map.get("Y", shape[-2] if len(shape) >= 2 else shape[-1])
                    )
                    if idx == 0:
                        base_w, base_h = width, height
                    downsample = 1.0
                    if base_w and width:
                        downsample = max(base_w / width, 1.0)
                    levels_info.append(
                        WSILevelMetadata(
                            level=idx,
                            dimensions=(width, height),
                            downsample=float(downsample),
                        )
                    )
                w, h = levels_info[0].dimensions
            else:
                arr, axes = self._read_level_region(file_path, level=0)
                image = self._collapse_to_image(arr, axes)
                h, w = int(image.shape[0]), int(image.shape[1])
                levels_info = [
                    WSILevelMetadata(level=0, dimensions=(w, h), downsample=1.0)
                ]

        return WSIContainer(
            slide_id="Xenium_Paired_Breast",
            file_path=file_path,
            dimensions_level_0=(w, h),
            mpp=self.mpp,
            staining="H&E_Registered",
            levels_count=len(levels_info),
            levels_info=levels_info,
            thumbnail_url=None,
        )

    def load_region(
        self,
        container: WSIContainer,
        x: int,
        y: int,
        width: int,
        height: int,
        level: int = 0,
    ) -> WSIRegionContainer:
        arr, axes = self._read_level_region(
            container.file_path,
            level=level,
            x=x,
            y=y,
            width=width,
            height=height,
        )
        crop_image = self._collapse_to_image(arr, axes)
        crop_rgb = self._ensure_rgb_uint8(crop_image)
        tensor = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return WSIRegionContainer(
            slide_id=container.slide_id,
            image_tensor=tensor,
            x_wsi=int(x),
            y_wsi=int(y),
            width_wsi=int(width),
            height_wsi=int(height),
            extracted_level=level,
            mpp=self.mpp,
        )

    def get_cells_in_region(
        self, x: int, y: int, width: int, height: int, use_he: bool = False
    ) -> List[CellFeaturePassport]:
        # Если use_he=True, фильтруем по вычисленным H&E координатам
        if use_he and "x_he_px" in self.cells_df.columns:
            mask = (
                (self.cells_df["x_he_px"] >= x)
                & (self.cells_df["x_he_px"] < x + width)
                & (self.cells_df["y_he_px"] >= y)
                & (self.cells_df["y_he_px"] < y + height)
            )
        else:
            x_um, y_um = x * self.mpp, y * self.mpp
            w_um, h_um = width * self.mpp, height * self.mpp
            mask = (
                (self.cells_df["x_centroid"] >= x_um)
                & (self.cells_df["x_centroid"] < x_um + w_um)
                & (self.cells_df["y_centroid"] >= y_um)
                & (self.cells_df["y_centroid"] < y_um + h_um)
            )

        subset = self.cells_df[mask]
        passports = []
        for _, row in subset.iterrows():
            m = self.patch_margin
            cx = float(row["x_centroid"]) / self.mpp
            cy = float(row["y_centroid"]) / self.mpp

            points = row["polygon_points"]
            poly_points = (
                [(px / self.mpp, py / self.mpp) for px, py in points]
                if isinstance(points, list)
                else []
            )

            poly = PolygonModel(points=poly_points)
            segmentation = SegmentationCellInstance(
                polygon=poly,
                bbox=(int(cx - m), int(cy - m), int(m * 2), int(m * 2)),
                score=1.0,
                centroid=(cx, cy),
            )
            passports.append(
                CellFeaturePassport(
                    slide_id="Xenium_Paired_Breast",
                    cell_id=row["cell_id"],
                    segmentation_data=segmentation,
                )
            )
        return passports

    def build_thumbnail_bytes(
        self, file_path: str, max_size: int = 1600, image_format: str = "JPEG"
    ) -> bytes:
        container = self.load_slide_metadata(file_path)
        selected_level = self._pick_level_for_downsample(
            container,
            desired_downsample=max(
                float(container.dimensions_level_0[0]) / float(max_size),
                float(container.dimensions_level_0[1]) / float(max_size),
            ),
        )
        arr, axes = self._read_level_region(file_path, level=selected_level)
        image_data = self._collapse_to_image(arr, axes)
        rgb = self._ensure_rgb_uint8(image_data)
        image = Image.fromarray(rgb)
        image.thumbnail((max_size, max_size))
        buffer = BytesIO()
        image.save(buffer, format=image_format, quality=85)
        return buffer.getvalue()

    def build_patch_bytes(
        self,
        file_path: str,
        x: int,
        y: int,
        width: int,
        height: int,
        max_size: int = 1024,
        image_format: str = "JPEG",
    ) -> bytes:
        container = self.load_slide_metadata(file_path)
        desired_downsample = max(
            float(width) / float(max_size), float(height) / float(max_size), 1.0
        )
        selected_level = self._pick_level_for_downsample(container, desired_downsample)
        selected_meta = container.levels_info[selected_level]
        ds = max(float(selected_meta.downsample), 1.0)
        level_x = int(x / ds)
        level_y = int(y / ds)
        level_width = max(1, int(np.ceil(width / ds)))
        level_height = max(1, int(np.ceil(height / ds)))
        arr, axes = self._read_level_region(
            file_path,
            level=selected_level,
            x=level_x,
            y=level_y,
            width=level_width,
            height=level_height,
        )
        image_data = self._collapse_to_image(arr, axes)
        rgb = self._ensure_rgb_uint8(image_data)
        image = Image.fromarray(rgb)
        image.thumbnail((max_size, max_size))
        buffer = BytesIO()
        image.save(buffer, format=image_format, quality=90)
        return buffer.getvalue()

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

        # --- КЛЮЧЕВОЙ ФИКС ---
        # Нормализуем cell_id в ОБОИХ датафреймах до того, как строим mapping.
        self.cells_df["cell_id"] = self.cells_df["cell_id"].apply(_normalize_cell_id)
        self.boundaries_df["cell_id"] = self.boundaries_df["cell_id"].apply(_normalize_cell_id)

        grouped = self.boundaries_df.groupby("cell_id")[["vertex_x", "vertex_y"]].apply(
            lambda x: list(zip(x["vertex_x"], x["vertex_y"]))
        )
        self.cells_df["polygon_points"] = self.cells_df["cell_id"].map(grouped)

        self.mpp = 0.212

        # --- Диагностика: сколько клеток реально получили полигон ---
        matched = self.cells_df["polygon_points"].apply(
            lambda p: isinstance(p, list) and len(p) > 0
        ).sum()
        total = len(self.cells_df)
        print(f"🔍 Полигоны границ найдены для {matched}/{total} клеток")
        if matched == 0:
            print(
                "⚠️ ВНИМАНИЕ: ни одна клетка не получила полигон из cell_boundaries.parquet. "
                "Проверь, что cells.parquet и cell_boundaries.parquet — из одного "
                "и того же экспорта Xenium (один run, одни и те же cell_id)."
            )
        elif matched < total:
            print(f"⚠️ {total - matched} клеток без полигона (нет в cell_boundaries.parquet)")

        # --- Диагностика системы координат ---
        # x_centroid/y_centroid и vertex_x/vertex_y у Xenium всегда в МИКРОНАХ.
        # Если get_cells_in_region() вызывается с x/y/width/height в ПИКСЕЛЯХ
        # морфологического изображения — нужно конвертировать через self.mpp,
        # иначе фильтр региона работает в другом масштабе, чем картинка.
        cx_max = self.cells_df["x_centroid"].max()
        cy_max = self.cells_df["y_centroid"].max()
        print(f"🔍 Диапазон центроидов (микроны): x<= {cx_max:.1f}, y<= {cy_max:.1f}")
        print(f"🔍 При mpp={self.mpp} это ~{cx_max/self.mpp:.0f} x {cy_max/self.mpp:.0f} px морфологии")

    def load_slide_metadata(self, file_path: str) -> WSIContainer:
        # x_centroid/y_centroid в cells_df — в МИКРОНАХ.
        # dimensions_level_0 должен быть в ПИКСЕЛЯХ, т.к. именно в пикселях
        # работает load_region() (crop по tiff-массиву).
        w_max = int(self.cells_df["x_centroid"].max() / self.mpp) + 1000
        h_max = int(self.cells_df["y_centroid"].max() / self.mpp) + 1000

        return WSIContainer(
            slide_id="Xenium_Paired_Breast",
            file_path=file_path,
            dimensions_level_0=(w_max, h_max),
            mpp=self.mpp,
            staining="Xenium_DAPI_Overlay",
            levels_count=1,
            levels_info=[WSILevelMetadata(level=0, dimensions=(w_max, h_max), downsample=1.0)],
        )

    def load_region(self, container: WSIContainer, x: int, y: int, width: int, height: int, level: int = 0) -> WSIRegionContainer:
        with tifffile.TiffFile(container.file_path) as tif:
            series = tif.series[0]
            crop = series.asarray(level=level)

        if crop.ndim > 3:
            crop = crop.squeeze()

        if crop.ndim == 3 and crop.shape[0] <= 4:
            crop = np.moveaxis(crop, 0, -1)

        h, w = crop.shape[:2]
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        width = min(width, w - x)
        height = min(height, h - y)

        crop_region = crop[y:y + height, x:x + width]

        if crop_region.ndim == 2:
            crop_rgb = np.stack([crop_region] * 3, axis=-1)
        elif crop_region.shape[-1] == 3:
            crop_rgb = crop_region
        else:
            crop_rgb = crop_region[..., :3]

        crop_rgb = crop_rgb.astype(np.float32)

        if crop_rgb.max() > 1.0:
            p2, p98 = np.percentile(crop_rgb, (2, 98))
            if p98 - p2 > 0:
                crop_rgb = np.clip((crop_rgb - p2) / (p98 - p2), 0, 1)
            else:
                crop_rgb = crop_rgb / crop_rgb.max()

        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).to(torch.float32)

        return WSIRegionContainer(
            slide_id=container.slide_id,
            image_tensor=tensor,
            x_wsi=int(x), y_wsi=int(y),
            width_wsi=int(width), height_wsi=int(height),
            extracted_level=level,
            mpp=self.mpp,
        )

    def get_cells_in_region(self, x: int, y: int, width: int, height: int) -> List[CellFeaturePassport]:
        # x, y, width, height приходят в ПИКСЕЛЯХ (та же система, что и load_region).
        # cells_df хранит координаты в МИКРОНАХ -> переводим bbox в микроны для фильтра.
        x_um = x * self.mpp
        y_um = y * self.mpp
        w_um = width * self.mpp
        h_um = height * self.mpp

        mask = (
            (self.cells_df["x_centroid"] >= x_um) & (self.cells_df["x_centroid"] < x_um + w_um) &
            (self.cells_df["y_centroid"] >= y_um) & (self.cells_df["y_centroid"] < y_um + h_um)
        )
        subset = self.cells_df[mask]

        passports = []
        for _, row in subset.iterrows():
            m = self.patch_margin
            # центроид и полигон переводим обратно в ПИКСЕЛИ, чтобы они были
            # в той же системе координат, что и region.x_wsi/y_wsi/image_tensor
            cx = float(row["x_centroid"]) / self.mpp
            cy = float(row["y_centroid"]) / self.mpp

            points = row["polygon_points"]
            if isinstance(points, list) and len(points) > 0:
                poly_points = [(px / self.mpp, py / self.mpp) for px, py in points]
            else:
                poly_points = []

            poly = PolygonModel(points=poly_points)

            segmentation = SegmentationCellInstance(
                polygon=poly,
                bbox=(int(cx - m), int(cy - m), int(m * 2), int(m * 2)),
                score=1.0,
                centroid=(cx, cy),
            )
            passports.append(CellFeaturePassport(
                slide_id="Xenium_Paired_Breast",
                cell_id=row["cell_id"],  # уже нормализован при загрузке
                segmentation_data=segmentation,
            ))
        return passports
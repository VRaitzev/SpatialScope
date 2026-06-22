"""
app/core/loaders/xenium_boundary_adapter.py
Реализация адаптера загрузки нативных и зарегистрированных границ Xenium.
"""

from typing import Any, Optional

import numpy as np

from app.core.loaders.rna import XeniumGenePanelLoader
from app.core.loaders.wsi.image import XeniumWSILoader
from app.core.loaders.wsi.segmentatioin.AbstractCellBoundaryLoader import (
    AbstractCellBoundaryLoader,
)
from app.core.schemes.Segmentation import (
    PolygonModel,
    SegmentationCellInstance,
    SegmentationContainer,
    SegmentationFeatureMeta,
    SegmentationFeatureValue,
    SegmentationModelManifest,
)
from app.core.schemes.WSI import WSIRegionContainer


class XeniumCellBoundaryAdapter(AbstractCellBoundaryLoader):
    def __init__(
        self, xenium_loader: XeniumWSILoader, gene_loader: XeniumGenePanelLoader
    ):
        self.loader = xenium_loader
        self.gene_loader = gene_loader

    @property
    def manifest(self) -> SegmentationModelManifest:
        return SegmentationModelManifest(
            model_name="Xenium_Onboard_Native_Loader",
            model_version="2.0",
            model_description="Data Adapter for Xenium native cell boundaries with runtime coordinate warp support.",
            supported_features=[
                SegmentationFeatureMeta(
                    feature_name="xenium_panel_expression",
                    description="Vector of 313 panel genes from counts matrix",
                    unit="counts",
                )
            ],
        )

    def load_boundaries(
        self, patch: WSIRegionContainer, registration_service: Optional[Any] = None
    ) -> SegmentationContainer:

        use_he = registration_service is not None
        cells_registry = self.loader.get_cells_in_region(
            patch.x_wsi, patch.y_wsi, patch.width_wsi, patch.height_wsi, use_he=use_he
        )

        valid_cells = []
        all_centroids = []
        all_polygon_points = []
        polygon_lengths = []

        # Шаг 1: Фильтрация и сбор данных
        for cell in cells_registry:
            gene_vector = self.gene_loader.get_gene_vector(cell.cell_id)
            if gene_vector is None:
                continue

            poly_pts = np.array(cell.segmentation_data.polygon.points)
            centroid_coord = cell.segmentation_data.centroid

            if poly_pts.size == 0 or poly_pts.ndim != 2 or poly_pts.shape[0] < 3:
                continue
            if centroid_coord is None or len(centroid_coord) < 2:
                continue

            valid_cells.append((cell, gene_vector))
            all_centroids.append(centroid_coord)
            all_polygon_points.append(poly_pts)
            polygon_lengths.append(poly_pts.shape[0])

        if not valid_cells:
            return SegmentationContainer(
                instances=[],
                image_shape=(patch.width_wsi, patch.height_wsi),
                model_name="Xenium_Empty",
                version="2.0",
            )

        # Шаг 2: Массовая трансформация координат (xenium-native -> he, либо identity)
        if registration_service is not None:
            centroids_arr = np.array(all_centroids, dtype=np.float64)
            warped_centroids = registration_service.warp_xenium_pixels_to_he_pixels(
                centroids_arr
            )

            flat_polygons = np.vstack(all_polygon_points)
            warped_flat_polygons = registration_service.warp_xenium_pixels_to_he_pixels(
                flat_polygons
            )

            warped_polygons = []
            curr_idx = 0
            for length in polygon_lengths:
                warped_polygons.append(
                    warped_flat_polygons[curr_idx : curr_idx + length]
                )
                curr_idx += length
        else:
            warped_centroids = np.array(all_centroids, dtype=np.float64)
            warped_polygons = [
                np.asarray(p, dtype=np.float64) for p in all_polygon_points
            ]

        # Шаг 2.5: ГЛАВНЫЙ ФИКС.
        # До этого момента координаты — АБСОЛЮТНЫЕ пиксели всего слайда
        # (что в native-режиме, что после warp в HE-пространство).
        # А patch.image_tensor — это кроп размера width_wsi x height_wsi,
        # начинающийся в точке (patch.x_wsi, patch.y_wsi).
        # Без вычитания смещения патча DinoV2ExpertAdapter получает координаты,
        # которые выходят далеко за пределы тензора патча, и тихо отбраковывает
        # все клетки (поэтому в логах "Получено 0 клеток").
        offset = np.array([patch.x_wsi, patch.y_wsi], dtype=np.float64)
        warped_centroids = warped_centroids - offset
        warped_polygons = [poly - offset for poly in warped_polygons]

        # Шаг 3: Сборка инстансов клеток
        instances = []
        for idx, (cell, gene_vector) in enumerate(valid_cells):
            poly_pts = warped_polygons[idx].copy()
            centroid_w = warped_centroids[idx]

            centroid = (float(centroid_w[0]), float(centroid_w[1]))

            min_x, min_y = np.min(poly_pts, axis=0)
            max_x, max_y = np.max(poly_pts, axis=0)
            bbox = (
                int(round(min_x)),
                int(round(min_y)),
                int(round(max_x)),
                int(round(max_y)),
            )

            poly_tuples = [tuple(map(float, p)) for p in poly_pts]
            polygon_model = PolygonModel(points=poly_tuples)

            feat_value = SegmentationFeatureValue(
                feature_name="xenium_panel_expression", value=gene_vector.tolist()
            )

            instances.append(
                SegmentationCellInstance(
                    polygon=polygon_model,
                    bbox=bbox,
                    score=cell.segmentation_data.score,
                    centroid=centroid,
                    features={"xenium_panel": feat_value},
                )
            )

        return SegmentationContainer(
            instances=instances,
            image_shape=(patch.width_wsi, patch.height_wsi),
            model_name="Xenium_Native"
            if registration_service is None
            else "Xenium_ANTsPy_Warped",
            version="2.0",
        )

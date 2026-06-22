import os
import pickle
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import KDTree
from tqdm import tqdm  

from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.schemes.Cells import ExpertModelManifest, FeatureMeta, WSICellFeatureRegistry, CellFeaturePassport
from app.core.schemes.Segmentation import SegmentationContainer

class GeometricMorphologyExpert(AbstractExpertModel):
    """
    [PRODUCTION LAYER] Детерминированный эксперт по геометрии и топологии.
    Считает форму клеток и плотность ткани. Оптимизирован кэшем и tqdm.
    """
    EXPERT_KEY = "Geometric_Morphology_Expert"

    def __init__(self, k_neighbors: int = 4, cache_dir: str = "./data/geometry_cache"):
        """
        Args:
            k_neighbors: Количество соседей для анализа плотности (включая себя).
            cache_dir: Путь к кэшу геометрических фич.
        """
        self.k_neighbors = k_neighbors
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def manifest(self) -> ExpertModelManifest:
        return ExpertModelManifest(
            model_name="Geometric_Morphology_Expert",
            model_version="v2_cached",
            model_description="Deterministic extractor of explicit shape and spatial density features with disk cache",
            supported_features=[
                FeatureMeta(feature_name="cell_area", description="Area enclosed by the cell polygon", unit="pixels^2"),
                FeatureMeta(feature_name="cell_perimeter", description="Total perimeter length of the cell boundary", unit="pixels"),
                FeatureMeta(feature_name="aspect_ratio", description="Ratio of major to minor axis of the minimum rotated bounding box", unit="ratio"),
                FeatureMeta(feature_name="solidity", description="Ratio of polygon area to its convex hull area (boundary roughness)", unit="ratio"),
                FeatureMeta(feature_name="mean_knn_distance", description="Average distance to the nearest k-3 neighboring cells", unit="pixels"),
                FeatureMeta(feature_name="local_density_index", description="Inverse of the mean distance to neighbors", unit="index")
            ]
        )

    def load_weights(self, path: str) -> None: pass
    def save_weights(self, path: str) -> None: pass

    def predict(self, cell_registry: SegmentationContainer, patch_id: str = "patch", slide_id: str = "xenium_slide") -> WSICellFeatureRegistry:
        """
        Расчет геометрических фич с дисковым кэшированием.
        """
        cache_path = os.path.join(self.cache_dir, f"{patch_id}_geometry.pkl")
        if os.path.exists(cache_path):
            print(f"        💾 [Кэш] {patch_id} (Геометрия) добавлен из файла.")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        instances = cell_registry.instances
        if not instances:
            return WSICellFeatureRegistry(slide_id=slide_id, cells=[], models_manifests=[self.manifest])

        centroids = [cell.centroid for cell in instances]
        centroids_np = np.array(centroids, dtype=np.float32)
        num_cells = len(instances)
        
        mean_knn_distances = np.zeros(num_cells, dtype=np.float32)
        density_indices = np.zeros(num_cells, dtype=np.float32)
        
        if num_cells > 1:
            tree = KDTree(centroids_np)
            current_k = min(self.k_neighbors, num_cells)
            distances, _ = tree.query(centroids_np, k=current_k)
            
            if current_k > 1:
                neighbor_distances = distances[:, 1:]  
                mean_knn_distances = np.mean(neighbor_distances, axis=1)
                density_indices = 1.0 / (mean_knn_distances + 1e-5)

        cells_passports = []
        
        for idx, cell in enumerate(tqdm(instances, desc=f"        📐 Расчет морфологии для {patch_id}", leave=False)):
            try:
                poly = cell.polygon.as_shapely()
            except Exception as e:
                continue

            area = 0.0
            perimeter = 0.0
            aspect_ratio = 1.0
            solidity = 1.0

            if poly.is_valid and poly.area > 0:
                area = poly.area
                perimeter = poly.length
                
                convex_area = poly.convex_hull.area
                if convex_area > 0:
                    solidity = area / convex_area
                    
                rotated_rect = poly.minimum_rotated_rectangle
                if rotated_rect.geom_type == 'Polygon':
                    rx, ry = rotated_rect.exterior.coords.xy
                    side_1 = float(np.hypot(rx[0] - rx[1], ry[0] - ry[1]))
                    side_2 = float(np.hypot(rx[1] - rx[2], ry[1] - ry[2]))
                    
                    major_axis = max(side_1, side_2)
                    minor_axis = min(side_1, side_2)
                    
                    if minor_axis > 0:
                        aspect_ratio = major_axis / minor_axis


            feature_vector = [
                float(area),
                float(perimeter),
                float(aspect_ratio),
                float(solidity),
                float(mean_knn_distances[idx]),
                float(density_indices[idx])
            ]

            cells_passports.append(CellFeaturePassport(
                cell_id=f"cell_{cell.centroid[0]}_{cell.centroid[1]}",
                slide_id=slide_id,
                segmentation_data=cell,
                expert_embeddings={self.EXPERT_KEY: feature_vector}
            ))

        registry = WSICellFeatureRegistry(
            slide_id=slide_id,
            x_wsi=0.0,
            y_wsi=0.0,
            x_local=0.0,
            y_local=0.0,
            cells=cells_passports,
            models_manifests=[self.manifest],
        )

        with open(cache_path, 'wb') as f:
            pickle.dump(registry, f)

        return registry
import os
import pickle
import logging  
import numpy as np
import cv2
import SimpleITK as sitk
import torch
from tqdm import tqdm  
from radiomics import featureextractor  
import radiomics
radiomics.logger.setLevel(logging.ERROR)  

from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.schemes.Cells import ExpertModelManifest, FeatureMeta, WSICellFeatureRegistry, CellFeaturePassport
from app.core.schemes.Segmentation import SegmentationContainer

import builtins  
for t in ['float', 'int', 'bool', 'object']:
    if not hasattr(np, t):
        setattr(np, t, getattr(builtins, t))  

class RadiomicsExpertAdapter(AbstractExpertModel):
    """
    [PRODUCTION LAYER] Оптимизированный эксперт радиомики.
    Логи глушатся, а прогресс выводится через аккуратный tqdm-бар.
    """
    EXPERT_KEY = "RadiomicsExpert"

    def __init__(self, cache_dir: str = "./data/radiomics_cache"):
        self.extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
        self.extractor.enableAllFeatures() 
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    @property
    def manifest(self) -> ExpertModelManifest:
        return ExpertModelManifest(
            model_name="Radiomics_Expert",
            model_version="v2_clean",
            model_description="Optimized radiomics extractor with clear progress bars",
            supported_features=[
                FeatureMeta(feature_name="radiomics_vec", description="Radiomics feature vector", unit=None)
            ]
        )

    def predict(self, patch_image: torch.Tensor, cell_registry: SegmentationContainer, patch_id: str = "patch") -> WSICellFeatureRegistry:
        cache_path = os.path.join(self.cache_dir, f"{patch_id}_radiomics.pkl")
        if os.path.exists(cache_path):
            print(f"        💾 [Кэш] {patch_id} загружен из файла.")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        cells_passports = []
        
        if isinstance(patch_image, torch.Tensor):
            np_image = patch_image.detach().cpu().numpy()
            if np_image.ndim == 3:
                np_image = np.moveaxis(np_image, 0, -1)
        else:
            np_image = np.array(patch_image)

        if np_image.ndim == 3 and np_image.shape[-1] == 3:
            np_image_gray = cv2.cvtColor((np_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            np_image_gray = (np_image * 255).astype(np.uint8)

        img_h, img_w = np_image_gray.shape
        
        instances = cell_registry.instances if cell_registry.instances else []
        
        for cell in tqdm(instances, desc=f"        📐 Расчет радиомики для {patch_id}", leave=False):
            points = cell.polygon.points if hasattr(cell, 'polygon') else []
            if not points:
                continue

            pts_array = np.array(points, dtype=np.int32)
            x_min, y_min = np.min(pts_array, axis=0)
            x_max, y_max = np.max(pts_array, axis=0)
            
            pad = 4
            x_min, y_min = max(0, int(x_min - pad)), max(0, int(y_min - pad))
            x_max, y_max = min(img_w, int(x_max + pad)), min(img_h, int(y_max + pad))
            
            if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                continue

            cropped_img = np_image_gray[y_min:y_max, x_min:x_max]
            local_pts = pts_array - [x_min, y_min]
            local_mask = np.zeros(cropped_img.shape, dtype=np.uint8)
            cv2.fillPoly(local_mask, [local_pts], 1)

            image_sitk = sitk.GetImageFromArray(cropped_img)
            mask_sitk = sitk.GetImageFromArray(local_mask)
            
            try:
                features = self.extractor.execute(image_sitk, mask_sitk)
                feat_vector = [float(v) for k, v in features.items() if isinstance(v, (int, float, np.number))]
            except Exception:
                feat_vector = [0.0] * 100 

            cells_passports.append(CellFeaturePassport(
                cell_id=f"cell_{cell.centroid[0]}_{cell.centroid[1]}",
                slide_id="xenium_slide",
                segmentation_data=cell,
                expert_embeddings={self.EXPERT_KEY: feat_vector}
            ))

        registry = WSICellFeatureRegistry(
            slide_id="xenium_slide", 
            x_wsi=0.0, y_wsi=0.0, x_local=0.0, y_local=0.0,  
            cells=cells_passports, 
            models_manifests=[self.manifest]
        )

        with open(cache_path, 'wb') as f:
            pickle.dump(registry, f)

        return registry

    def load_weights(self, path: str): pass
    def save_weights(self, path: str): pass
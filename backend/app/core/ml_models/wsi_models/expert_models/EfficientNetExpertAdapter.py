import os
import pickle
import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import roi_align
from tqdm import tqdm  

from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.schemes.Cells import ExpertModelManifest, FeatureMeta, WSICellFeatureRegistry, CellFeaturePassport
from app.core.schemes.Segmentation import SegmentationContainer

class EfficientNetExpertAdapter(AbstractExpertModel):
    """
    [PRODUCTION LAYER] Морфологический экстрактор на базе EfficientNet-B0.
    Оптимизирован с помощью пакетного дискового кэширования и tqdm-индикатора.
    """
    EXPERT_KEY = "EfficientNetExpert"

    def __init__(self, batch_size: int = 32, cache_dir: str = "./data/efficientnet_cache"):
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(self.device).eval()
        self.batch_size = batch_size

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @property
    def manifest(self) -> ExpertModelManifest:
        return ExpertModelManifest(
            model_name="EfficientNet_B0_Extractor",
            model_version="v2_cached",
            model_description="EfficientNet-B0 feature extractor with batch caching and clean tracking",
            supported_features=[
                FeatureMeta(feature_name="effnet_embedding", description="1280-dim latent vector", unit=None)
            ]
        )

    def predict(self, patch_image: torch.Tensor, cell_registry: SegmentationContainer, patch_id: str = "patch") -> WSICellFeatureRegistry:
        """
        Предикт с батчевым извлечением через roi_align и сохранением на диск.
        """
        cache_path = os.path.join(self.cache_dir, f"{patch_id}_effnet.pkl")
        if os.path.exists(cache_path):
            print(f"        💾 [Кэш] {patch_id} (EfficientNet) загружен из файла.")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        instances = cell_registry.instances
        if not instances:
            return WSICellFeatureRegistry(slide_id="unknown_slide", cells=[], models_manifests=[self.manifest])

        patch_tensor = patch_image.to(self.device).to(torch.float32) / 255.0
        patch_batch = patch_tensor.unsqueeze(0)
        _, patch_h, patch_w = patch_tensor.shape

        boxes = []
        valid_cells = []
        for cell in instances:
            x1, y1, x2, y2 = cell.bbox
            boxes.append([0.0, max(0.0, float(x1)), max(0.0, float(y1)), min(float(patch_w), float(x2)), min(float(patch_h), float(y2))])
            valid_cells.append(cell)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        cells_passports = []

        with torch.no_grad():
            for start in tqdm(
                range(0, len(valid_cells), self.batch_size), 
                desc=f"        🧠 Извлечение нейрофич для {patch_id}", 
                leave=False
            ):
                chunk_cells = valid_cells[start:start + self.batch_size]
                chunk_boxes = boxes_tensor[start:start + self.batch_size]

                crops = roi_align(patch_batch, chunk_boxes, output_size=(224, 224), aligned=True)
                crops = (crops - self.mean) / self.std

                embeddings = self.feature_extractor(crops).squeeze(-1).squeeze(-1)
                embeddings = embeddings.cpu().numpy().tolist()

                for cell, emb in zip(chunk_cells, embeddings):
                    cells_passports.append(CellFeaturePassport(
                        cell_id=f"cell_{cell.centroid[0]}_{cell.centroid[1]}",
                        slide_id="xenium_slide",
                        segmentation_data=cell,
                        expert_embeddings={self.EXPERT_KEY: emb}
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
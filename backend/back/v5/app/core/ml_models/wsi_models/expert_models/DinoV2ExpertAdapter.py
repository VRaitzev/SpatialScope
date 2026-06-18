import torch
from torchvision.ops import roi_align

from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.schemes.Cells import (
    ExpertModelManifest, FeatureMeta, WSICellFeatureRegistry,
    CellFeaturePassport
)
from app.core.schemes.Segmentation import SegmentationContainer


class DinoV2ExpertAdapter(AbstractExpertModel):
    """
    [MODEL LAYER] Экстракция морфологических признаков клеток через DINOv2 ViT-S/14.

    Оптимизация: вместо поячеечного F.interpolate в питоновском цикле все боксы
    клеток вырезаются и ресайзятся ОДНИМ батчевым вызовом torchvision.ops.roi_align.
    """

    # Ключ должен совпадать с тем, что ждёт XeniumGraphDataset
    EXPERT_KEY = "DinoV2ExpertAdapter"

    def __init__(self, batch_size: int = 64):
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        self.batch_size = batch_size

        print(
            "[DinoV2ExpertAdapter] device:", self.device,
            "| cuda available:", torch.cuda.is_available(),
            "| gpu name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @property
    def manifest(self) -> ExpertModelManifest:
        return ExpertModelManifest(
            model_name="DinoV2_Extractor",
            model_version="v1",
            model_description="DINOv2 ViT-S/14 morphological feature extractor",
            supported_features=[
                FeatureMeta(
                    feature_name="dinov2_embedding",
                    description="384-dimensional latent morphological vector",
                    unit=None
                )
            ]
        )

    def load_weights(self, path: str):
        pass

    def save_weights(self, path: str):
        pass

    def predict(
        self,
        patch_image: torch.Tensor,
        cell_registry: SegmentationContainer
    ) -> WSICellFeatureRegistry:
        instances = cell_registry.instances
        if not instances:
            return WSICellFeatureRegistry(slide_id="unknown_slide", cells=[], models_manifests=[self.manifest])

        patch_tensor = patch_image.to(self.device, non_blocking=True).to(torch.float32)
        if patch_tensor.max() > 1.0:
            patch_tensor = patch_tensor / 255.0
        patch_batch = patch_tensor.unsqueeze(0)  # (1, 3, H, W) — нужен для roi_align
        _, patch_h, patch_w = patch_tensor.shape

        # bbox хранится как (min_x, min_y, max_x, max_y), а не (x, y, w, h)!
        valid_cells = []
        boxes = []
        for cell in instances:
            x1, y1, x2, y2 = cell.bbox
            x1c, y1c = max(0.0, float(x1)), max(0.0, float(y1))
            x2c, y2c = min(float(patch_w), float(x2)), min(float(patch_h), float(y2))
            if x2c <= x1c or y2c <= y1c:
                continue
            boxes.append([0.0, x1c, y1c, x2c, y2c])  # 0 — индекс изображения в батче
            valid_cells.append(cell)

        if not valid_cells:
            return WSICellFeatureRegistry(slide_id="xenium_slide", cells=[], models_manifests=[self.manifest])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)

        cells_passports = []
        autocast_enabled = self.device.type == "cuda"

        with torch.no_grad():
            for start in range(0, len(valid_cells), self.batch_size):
                chunk_cells = valid_cells[start:start + self.batch_size]
                chunk_boxes = boxes_tensor[start:start + self.batch_size]

                crops = roi_align(
                    patch_batch, chunk_boxes,
                    output_size=(224, 224),
                    aligned=True,
                )  # -> (chunk_size, 3, 224, 224)

                crops = (crops - self.mean) / self.std

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=autocast_enabled):
                    embeddings = self.model(crops)

                embeddings = embeddings.float().cpu().numpy().tolist()

                for cell, emb in zip(chunk_cells, embeddings):
                    passport = CellFeaturePassport(
                        cell_id=f"cell_{cell.centroid[0]}_{cell.centroid[1]}",
                        slide_id="xenium_slide",
                        segmentation_data=cell,
                        expert_embeddings={self.EXPERT_KEY: emb}
                    )
                    cells_passports.append(passport)

        return WSICellFeatureRegistry(
            slide_id="xenium_slide",
            x_wsi=0.0, y_wsi=0.0, x_local=0.0, y_local=0.0,
            cells=cells_passports,
            models_manifests=[self.manifest]
        )
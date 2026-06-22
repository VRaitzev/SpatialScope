from typing import List, Optional

import torch

from app.core.ml_models.wsi_models.expert_models.AbstractExpertModel import AbstractExpertModel
from app.core.schemes.Cells import CellFeaturePassport, ExpertModelManifest, WSICellFeatureRegistry
from app.core.schemes.Segmentation import SegmentationContainer


class CompositeExpertAdapter(AbstractExpertModel):
    """Runs multiple expert models and concatenates their embeddings."""

    def __init__(
        self,
        experts: List[AbstractExpertModel],
        composite_key: str,
        image_expert_keys: Optional[List[str]] = None,
    ):
        self.experts = experts
        self.composite_key = composite_key
        self.image_expert_keys = set(image_expert_keys or ["EfficientNetExpert", "RadiomicsExpert"])

    @property
    def EXPERT_KEY(self) -> str:
        return self.composite_key

    @property
    def manifest(self) -> ExpertModelManifest:
        names = ", ".join(ex.manifest.model_name for ex in self.experts)
        return ExpertModelManifest(
            model_name=f"Composite[{names}]",
            model_version="v1",
            model_description="Concatenated multi-expert morphological profile",
            supported_features=[],
        )

    def load_weights(self, path: str) -> None:
        pass

    def save_weights(self, path: str) -> None:
        pass

    def _needs_image(self, expert: AbstractExpertModel) -> bool:
        key = getattr(expert, "EXPERT_KEY", "")
        return key in self.image_expert_keys

    def predict(
        self,
        patch_image: Optional[torch.Tensor] = None,
        cell_registry: SegmentationContainer = None,
        patch_id: str = "patch",
        slide_id: str = "xenium_slide",
    ) -> WSICellFeatureRegistry:
        merged: dict[str, CellFeaturePassport] = {}
        manifests = []

        for expert in self.experts:
            if self._needs_image(expert):
                if patch_image is None:
                    raise ValueError(f"Expert {expert.manifest.model_name} requires patch image")
                registry = expert.predict(
                    patch_image=patch_image,
                    cell_registry=cell_registry,
                    patch_id=patch_id,
                )
            else:
                registry = expert.predict(
                    cell_registry=cell_registry,
                    patch_id=patch_id,
                    slide_id=slide_id,
                )

            manifests.append(expert.manifest)
            for cell in registry.cells:
                if cell.cell_id not in merged:
                    merged[cell.cell_id] = cell.model_copy(deep=True)
                else:
                    merged[cell.cell_id].expert_embeddings.update(cell.expert_embeddings)

        ordered_keys = [getattr(ex, "EXPERT_KEY", "") for ex in self.experts]
        for cell in merged.values():
            concat: List[float] = []
            for key in ordered_keys:
                if key in cell.expert_embeddings:
                    vec = cell.expert_embeddings[key]
                    concat.extend([float(v) for v in vec])
            cell.expert_embeddings[self.composite_key] = concat

        return WSICellFeatureRegistry(
            slide_id=slide_id,
            x_wsi=0.0,
            y_wsi=0.0,
            x_local=0.0,
            y_local=0.0,
            cells=list(merged.values()),
            models_manifests=manifests,
        )

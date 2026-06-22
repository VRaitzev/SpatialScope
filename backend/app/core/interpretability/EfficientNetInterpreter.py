import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from app.core.interpretability.AbstractModelInterpreter import (
    AbstractModelInterpreter,
    CellAttributionMap,
    FeatureAttribution,
    InterpretationResult,
    SpatialAttributionMap,
)
from app.core.schemes.Cells import WSICellFeatureRegistry
from app.core.schemes.Segmentation import SegmentationContainer


class EfficientNetInterpreter(AbstractModelInterpreter):
    """GradCAM-style saliency for EfficientNet cell embeddings."""

    def __init__(self, expert_model):
        self.expert = expert_model
        self.device = getattr(expert_model, "device", torch.device("cpu"))

    @property
    def model_type(self) -> str:
        return "efficientnet"

    def _patch_tensor(self, patch_image: torch.Tensor) -> torch.Tensor:
        tensor = patch_image.to(self.device).float()
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def interpret(
        self,
        *,
        patch_image=None,
        cell_registry=None,
        spatial_model=None,
        target_gene: Optional[str] = None,
        patch_id: str = "patch",
    ) -> InterpretationResult:
        if patch_image is None or cell_registry is None:
            return InterpretationResult(
                model_type=self.model_type,
                model_name="EfficientNet_B0",
                summary="Недостаточно данных для интерпретации EfficientNet.",
            )

        patch_norm = self._patch_tensor(patch_image)
        patch_norm.requires_grad_(True)

        activations = []
        gradients = []

        def forward_hook(_module, _inp, output):
            activations.append(output)

        def backward_hook(_module, _grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = self.expert.feature_extractor[-1]
        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)

        try:
            features = self.expert.feature_extractor(patch_norm)
            score = features.mean()
            self.expert.feature_extractor.zero_grad(set_to_none=True)
            score.backward(retain_graph=True)

            if not activations or not gradients:
                raise RuntimeError("GradCAM hooks did not capture tensors")

            act = activations[0].detach()
            grad = gradients[0].detach()
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=patch_norm.shape[-2:], mode="bilinear", align_corners=False)
            cam_np = cam.squeeze().cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        finally:
            handle_f.remove()
            handle_b.remove()

        instances = cell_registry.instances if isinstance(cell_registry, SegmentationContainer) else []
        cell_maps: list[CellAttributionMap] = []

        for cell in instances:
            x1, y1, x2, y2 = [int(v) for v in cell.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(cam_np.shape[1], x2)
            y2 = min(cam_np.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                importance = 0.0
            else:
                importance = float(cam_np[y1:y2, x1:x2].mean())

            cell_maps.append(
                CellAttributionMap(
                    cell_id=f"cell_{cell.centroid[0]}_{cell.centroid[1]}",
                    centroid=[float(cell.centroid[0]), float(cell.centroid[1])],
                    importance_score=importance,
                    attributions=[
                        FeatureAttribution(
                            feature_name="gradcam_saliency",
                            importance=importance,
                            description="GradCAM saliency inside cell bbox",
                        )
                    ],
                )
            )

        cell_maps.sort(key=lambda c: c.importance_score, reverse=True)
        top = cell_maps[:5]

        return InterpretationResult(
            model_type=self.model_type,
            model_name="EfficientNet_B0",
            summary=(
                f"GradCAM saliency for patch {patch_id}"
                + (f", target gene context: {target_gene}" if target_gene else "")
            ),
            cell_maps=cell_maps,
            spatial_map=SpatialAttributionMap(
                width=int(cam_np.shape[1]),
                height=int(cam_np.shape[0]),
                values=cam_np.tolist(),
            ),
            top_features=[
                FeatureAttribution(
                    feature_name=f"cell_{idx + 1}",
                    importance=item.importance_score,
                    description=f"Cell at {item.centroid}",
                )
                for idx, item in enumerate(top)
            ],
            metadata={"patch_id": patch_id, "target_gene": target_gene},
        )

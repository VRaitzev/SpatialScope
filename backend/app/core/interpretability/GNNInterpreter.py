import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from app.core.interpretability.AbstractModelInterpreter import (
    AbstractModelInterpreter,
    CellAttributionMap,
    FeatureAttribution,
    InterpretationResult,
)
from app.core.schemes.Cells import WSICellFeatureRegistry


class GNNInterpreter(AbstractModelInterpreter):
    """Node-level attribution for spatial GNN using gradients and GAT attention."""

    def __init__(self, spatial_adapter):
        self.spatial = spatial_adapter
        self.device = getattr(spatial_adapter, "device", torch.device("cpu"))

    @property
    def model_type(self) -> str:
        return "gnn"

    def interpret(
        self,
        *,
        patch_image=None,
        cell_registry=None,
        spatial_model=None,
        target_gene: Optional[str] = None,
        patch_id: str = "patch",
    ) -> InterpretationResult:
        registry: WSICellFeatureRegistry = cell_registry
        if registry is None or not registry.cells:
            return InterpretationResult(
                model_type=self.model_type,
                model_name="Spatial_GAT",
                summary="Нет клеток для интерпретации GNN.",
            )

        graph = self.spatial._build_graph(registry)
        graph.x = graph.x.detach().clone().requires_grad_(True)
        graph = graph.to(self.device)
        model = self.spatial.model
        model.eval()

        edge_weights = []
        x1, edge_index1 = graph.x, graph.edge_index
        out1, (edge_index_att, att_weights) = model.conv1(x1, edge_index1, return_attention_weights=True)
        edge_weights.append(att_weights.detach().cpu().mean(dim=1).numpy())

        out = model.conv2(F.elu(out1), edge_index1)
        if target_gene and target_gene in self.spatial.gene_names:
            gene_idx = self.spatial.gene_names.index(target_gene)
        else:
            gene_idx = 0

        target = out[:, gene_idx].sum()
        model.zero_grad(set_to_none=True)
        if graph.x.grad is not None:
            graph.x.grad.zero_()
        target.backward()

        node_grad = graph.x.grad.detach().cpu().numpy()
        node_importance = np.abs(node_grad).mean(axis=1)

        if edge_weights:
            src = edge_index_att[0].cpu().numpy()
            att = edge_weights[0]
            node_edge_score = np.zeros(len(registry.cells), dtype=np.float32)
            counts = np.zeros(len(registry.cells), dtype=np.float32)
            for idx, s in enumerate(src):
                if s < len(node_edge_score):
                    node_edge_score[s] += float(att[idx])
                    counts[s] += 1.0
            counts[counts == 0] = 1.0
            node_edge_score /= counts
            combined = 0.7 * node_importance + 0.3 * node_edge_score
        else:
            combined = node_importance

        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        cell_maps: list[CellAttributionMap] = []
        for idx, cell in enumerate(registry.cells):
            feat_attr = []
            if graph.x.grad is not None:
                grad_row = np.abs(node_grad[idx])
                top_idx = np.argsort(grad_row)[-3:][::-1]
                for fi in top_idx:
                    feat_attr.append(
                        FeatureAttribution(
                            feature_name=f"feature_{int(fi)}",
                            importance=float(grad_row[fi]),
                            description="Input feature gradient magnitude",
                        )
                    )

            cell_maps.append(
                CellAttributionMap(
                    cell_id=cell.cell_id,
                    centroid=[
                        float(cell.segmentation_data.centroid[0]),
                        float(cell.segmentation_data.centroid[1]),
                    ],
                    importance_score=float(combined[idx]),
                    attributions=feat_attr,
                )
            )

        cell_maps.sort(key=lambda c: c.importance_score, reverse=True)
        gene_label = target_gene or self.spatial.gene_names[gene_idx]

        return InterpretationResult(
            model_type=self.model_type,
            model_name="Spatial_GAT",
            summary=f"GNN node attribution for gene {gene_label} on patch {patch_id}",
            cell_maps=cell_maps,
            top_features=[
                FeatureAttribution(
                    feature_name=item.cell_id,
                    importance=item.importance_score,
                    description=f"Centroid {item.centroid}",
                )
                for item in cell_maps[:5]
            ],
            metadata={
                "patch_id": patch_id,
                "target_gene": gene_label,
                "expert_key": self.spatial.expert_key,
            },
        )

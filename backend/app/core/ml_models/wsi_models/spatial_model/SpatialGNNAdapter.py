import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay, KDTree
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops
from typing import List, Optional

from app.core.ml_models.wsi_models.spatial_model.AbstarctSpatialModel import AbstractSpatialSpotModel
from app.core.schemes.Cells import (
    WSICellFeatureRegistry, RegionalSpatialSpot, 
    RegionalPredictionOutput, SpotExpressionPrediction,
    ExpertModelManifest
)

class TranscriptomicsGNN(torch.nn.Module):
    """
    Классическая архитектура Graph Attention Network для предсказания экспрессии.
    """
    def __init__(self, in_channels: int, hidden_dim: int, gene_count: int):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_dim, heads=4)
        self.conv2 = GATv2Conv(hidden_dim * 4, gene_count, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x


class SpatialGNNAdapter(AbstractSpatialSpotModel):
    """
    [MODEL LAYER] Адаптер для графовой сети. 
    Строит пространственный граф клеток и предсказывает экспрессию генов.
    """
    def __init__(
        self, 
        feature_dim: int, 
        hidden_dim: int, 
        gene_names: List[str], 
        expert_key: str,
        spot_radius: float = 50.0
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gene_names = gene_names
        self.expert_key = expert_key
        self.spot_radius = spot_radius
        self.hidden_dim = hidden_dim
        self._feature_dim = feature_dim
        self.model = TranscriptomicsGNN(feature_dim, hidden_dim, len(gene_names))
        self.model.to(self.device)

    def _ensure_model(self, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.model is not None and self._feature_dim == feature_dim:
            return
        self._feature_dim = feature_dim
        self.model = TranscriptomicsGNN(feature_dim, self.hidden_dim, len(self.gene_names))
        self.model.to(self.device)

    @property
    def manifest(self) -> ExpertModelManifest:
        """Метаданные модели для реестра системы."""
        return ExpertModelManifest(
            model_name="Spatial_GAT_Transcriptomics_Predictor",
            model_version="v2",
            model_description="Predicts spatial gene expression from cell geometry and tissue topology using GATv2",
            supported_features=[]
        )

    def load_weights(self, path: str) -> None:
        """Load trained GNN weights."""
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        first_key = next(iter(state_dict))
        if "conv1" in first_key and "weight" in first_key:
            weight = state_dict[first_key]
            if weight.ndim >= 2:
                self._ensure_model(int(weight.shape[1]))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def save_weights(self, path: str) -> None:
        """Сохранение весов GNN."""
        torch.save(self.model.state_dict(), path)

    def _build_graph(self, patch_registry: WSICellFeatureRegistry) -> Data:
        """
        Строит граф на основе триангуляции Делоне и наполняет его динамическими фичами эксперта.
        """
        if not patch_registry.cells:
            raise ValueError("Реестр клеток пуст. Невозможно построить граф.")

        # 1. Извлекаем координаты центроидов клеток
        coords = np.array([c.segmentation_data.centroid for c in patch_registry.cells], dtype=np.float32)
        
        # 2. Триангуляция Делоне
        edges = set()
        if len(coords) >= 3:
            tri = Delaunay(coords)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        u, v = simplex[i], simplex[j]
                        edges.add((u, v))
                        edges.add((v, u))

        if edges:
            edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Добавляем петли к самим себе
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(coords))
        
        # 3. Динамическое извлечение фич по переданному ключу эксперта
        x_list = []
        for c in patch_registry.cells:
            if self.expert_key not in c.expert_embeddings:
                raise KeyError(
                    f"Expert key '{self.expert_key}' missing for cell {c.cell_id}. "
                    f"Available: {list(c.expert_embeddings.keys())}"
                )
            x_list.append(c.expert_embeddings[self.expert_key])

        feature_dim = len(x_list[0])
        self._ensure_model(feature_dim)
        x = torch.tensor(x_list, dtype=torch.float, device=self.device)
        edge_index = edge_index.to(self.device)
        
        return Data(x=x, edge_index=edge_index)

    def predict_spatial_expression(
        self, 
        patch_registry: WSICellFeatureRegistry, 
        regional_spots: List[RegionalSpatialSpot],
        patch_id: str = "unknown_patch"  # Передаем patch_id динамически
    ) -> RegionalPredictionOutput:
        """
        Основной метод инференса: строит граф, прогоняет через GAT и агрегирует в споты.
        """
        self.model.eval()
        
        if not patch_registry.cells:
            return RegionalPredictionOutput(patch_id=patch_id, predictions=[])

        with torch.no_grad():
            # Шаг 1: Сборка графа и предсказание
            graph = self._build_graph(patch_registry)
            cell_preds = self.model(graph.x, graph.edge_index).cpu().numpy()
            
            # Шаг 2: Геометрический маппинг клеток на споты через KDTree
            cell_coords = np.array([c.segmentation_data.centroid for c in patch_registry.cells], dtype=np.float32)
            tree = KDTree(cell_coords)
            
            predictions = []
            
            for spot in regional_spots:
                # Используем динамический радиус из self.spot_radius
                indices = tree.query_ball_point([spot.x_local, spot.y_local], self.spot_radius)
                
                if indices:
                    spot_values = cell_preds[indices].mean(axis=0)
                    expr_map = {name: float(val) for name, val in zip(self.gene_names, spot_values)}
                else:
                    expr_map = {name: 0.0 for name in self.gene_names}
                
                predictions.append(SpotExpressionPrediction(
                    spot_id=spot.spot_id,
                    predicted_expressions=expr_map
                ))
            
            return RegionalPredictionOutput(patch_id=patch_id, predictions=predictions)
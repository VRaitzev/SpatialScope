import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import Delaunay, KDTree
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops

from app.core.ml_models.wsi_models import AbstractSpatialSpotModel
from app.core.schemes.Cells import (
    WSICellFeatureRegistry, RegionalSpatialSpot, 
    RegionalPredictionOutput, SpotExpressionPrediction
)

class TranscriptomicsGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, gene_count):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_dim, heads=4)
        self.conv2 = GATv2Conv(hidden_dim * 4, gene_count)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class SpatialGNNAdapter(AbstractSpatialSpotModel):
    def __init__(self, feature_dim=384, hidden_dim=64, gene_count=10):
        self.model = TranscriptomicsGNN(feature_dim, hidden_dim, gene_count)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.gene_names = [f"GENE_{i}" for i in range(gene_count)]

    def _build_graph(self, patch_registry: WSICellFeatureRegistry) -> Data:
        coords = np.array([c.segmentation_data.centroid for c in patch_registry.cells])
        # Триангуляция Делоне для честной топологии
        tri = Delaunay(coords)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = simplex[i], simplex[j]
                    edges.add(tuple(sorted((u, v))))
        
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(coords))
        
        x = torch.tensor([c.expert_embeddings["DinoV2_Extractor"] for c in patch_registry.cells], dtype=torch.float)
        return Data(x=x, edge_index=edge_index.to(self.device))

    def predict_spatial_expression(self, patch_registry, regional_spots) -> RegionalPredictionOutput:
        self.model.eval()
        with torch.no_grad():
            graph = self._build_graph(patch_registry)
            # Прямой проход: [num_cells, gene_count]
            cell_preds = self.model(graph.x, graph.edge_index).cpu().numpy()
            
            # Подготовка KDTree для быстрого поиска клеток внутри спота
            cell_coords = np.array([c.segmentation_data.centroid for c in patch_registry.cells])
            tree = KDTree(cell_coords)
            
            predictions = []
            # Радиус спота (в пикселях). Обычно около 25-50 мкм
            spot_radius = 50.0 
            
            for spot in regional_spots:
                # Находим все клетки в радиусе спота
                indices = tree.query_ball_point([spot.x_local, spot.y_local], spot_radius)
                
                if indices:
                    # Агрегация: усредняем предсказания всех найденных клеток
                    spot_values = cell_preds[indices].mean(axis=0)
                    expr_map = {name: float(val) for name, val in zip(self.gene_names, spot_values)}
                else:
                    # Если клеток нет, отдаем нули
                    expr_map = {name: 0.0 for name in self.gene_names}
                
                predictions.append(SpotExpressionPrediction(
                    spot_id=spot.spot_id,
                    predicted_expressions=expr_map
                ))
            
            return RegionalPredictionOutput(patch_id="patch_01", predictions=predictions)

    @property
    def manifest(self): return ...
    def load_weights(self, path): pass
    def save_weights(self, path): pass
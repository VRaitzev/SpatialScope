import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from scipy.spatial import Delaunay
from torch_geometric.utils import add_self_loops
import numpy as np
from typing import List
from app.core.schemes.Cells import WSICellFeatureRegistry

class XeniumGraphDataset(Dataset):
    """
    [DATA LAYER] Высокопроизводительный датасет для обучения GNN с динамической поддержкой экспертов.
    """
    def __init__(
        self, 
        registries: List[WSICellFeatureRegistry], 
        target_gene_names: List[str],
        all_gene_names: List[str] = None,
        expert_key: str = "EfficientNetExpert",
        min_cells_per_graph: int = 5  
    ):
        self.registries = registries
        self.target_gene_names = target_gene_names
        self.all_gene_names = all_gene_names
        self.expert_key = expert_key
        self.min_cells_per_graph = min_cells_per_graph
        self._precomputed_graphs = []
        
        self.gene_to_idx = {name: idx for idx, name in enumerate(all_gene_names)} if all_gene_names else {}
        
        print(f"⚙️ [Dataset Init] Режим: {expert_key} | Целевых генов: {len(target_gene_names)}")
        self._prepare_data()

    def _prepare_data(self):
        for idx, registry in enumerate(self.registries):
            if registry is None or not registry.cells:
                continue
                
            num_cells = len(registry.cells)
            
            if num_cells < self.min_cells_per_graph:
                print(f"⚠️ [Dataset] Регистр #{idx} содержит только {num_cells} клеток (меньше {self.min_cells_per_graph}) — пропускаем")
                continue
                
            try:
                coords = np.array([c.segmentation_data.centroid for c in registry.cells], dtype=np.float32)
                
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
                    edge_index = torch.tensor([[i, i] for i in range(num_cells)], dtype=torch.long).t().contiguous()
                    
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_cells)

                x_features = []
                
                target_dim = None
                for c in registry.cells:
                    if self.expert_key in c.expert_embeddings:
                        emb = c.expert_embeddings[self.expert_key]
                        if emb is not None and hasattr(emb, '__len__') and len(emb) > 0:
                            current_len = len(emb)
                            if target_dim is None or current_len > target_dim:
                                target_dim = current_len
                
                if target_dim is None:
                    target_dim = 100
                    print(f"⚠️ [Dataset] Эмбеддинги для {self.expert_key} не найдены, использую размерность {target_dim}")

                for c in registry.cells:
                    if self.expert_key not in c.expert_embeddings:
                        feat = [0.0] * target_dim
                    else:
                        feat = c.expert_embeddings[self.expert_key]
                        
                        if isinstance(feat, torch.Tensor):
                            feat = feat.detach().cpu().tolist()
                        elif isinstance(feat, np.ndarray):
                            feat = feat.tolist()
                        else:
                            feat = list(feat)
                        
                        if len(feat) != target_dim:
                            if len(feat) < target_dim:
                                feat = feat + [0.0] * (target_dim - len(feat))
                            else:
                                feat = feat[:target_dim]
                                
                    x_features.append(feat)
                    
                x_tensor = torch.tensor(x_features, dtype=torch.float)

                y_features = []
                for c in registry.cells:
                    cell_genes_data = None
                    if hasattr(c.segmentation_data, 'features') and c.segmentation_data.features:
                        features_dict = c.segmentation_data.features
                        if isinstance(features_dict, dict) and "xenium_panel" in features_dict:
                            cell_genes_data = features_dict["xenium_panel"].value
                    
                    if cell_genes_data is None:
                        gene_values = [0.0] * len(self.target_gene_names)
                    else:
                        if isinstance(cell_genes_data, np.ndarray):
                            cell_genes_data = cell_genes_data.tolist()
                        
                        if isinstance(cell_genes_data, dict):
                            gene_values = [float(cell_genes_data.get(gene, 0.0)) for gene in self.target_gene_names]
                        elif isinstance(cell_genes_data, list):
                            if not self.all_gene_names:
                                raise ValueError("Клетка хранит экспрессию списком, но список всех генов (all_gene_names) не передан в датасет!")
                            gene_values = []
                            for gene in self.target_gene_names:
                                g_idx = self.gene_to_idx.get(gene)
                                gene_values.append(float(cell_genes_data[g_idx]) if g_idx is not None and g_idx < len(cell_genes_data) else 0.0)
                        else:
                            try:
                                gene_values = [float(x) for x in cell_genes_data[:len(self.target_gene_names)]]
                            except:
                                gene_values = [0.0] * len(self.target_gene_names)
                    
                    if len(gene_values) != len(self.target_gene_names):
                        if len(gene_values) < len(self.target_gene_names):
                            gene_values = gene_values + [0.0] * (len(self.target_gene_names) - len(gene_values))
                        else:
                            gene_values = gene_values[:len(self.target_gene_names)]
                    
                    y_features.append(gene_values)
                    
                y_tensor = torch.tensor(y_features, dtype=torch.float)

                graph_data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
                self._precomputed_graphs.append(graph_data)

            except Exception as e:
                print(f"❌ Ошибка сборки графа на регистре #{idx}: {str(e)}")
                import traceback
                traceback.print_exc()

    def __len__(self): 
        return len(self._precomputed_graphs)
    
    def __getitem__(self, idx): 
        return self._precomputed_graphs[idx]
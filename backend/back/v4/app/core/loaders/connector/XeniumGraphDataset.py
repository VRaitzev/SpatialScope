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
        expert_key: str = "EfficientNetExpert"
    ):
        self.registries = registries
        self.target_gene_names = target_gene_names
        self.all_gene_names = all_gene_names
        self.expert_key = expert_key
        self._precomputed_graphs = []
        
        # Индексация для быстрого поиска гена в массиве
        self.gene_to_idx = {name: idx for idx, name in enumerate(all_gene_names)} if all_gene_names else {}
        
        print(f"⚙️ [Dataset Init] Режим: {expert_key} | Целевых генов: {len(target_gene_names)}")
        self._prepare_data()

    def _prepare_data(self):
        for idx, registry in enumerate(self.registries):
            if registry is None or not registry.cells:
                continue
                
            num_cells = len(registry.cells)
            
            try:
                # 1. Топология графа (Делоне)
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
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    
                edge_index, _ = add_self_loops(edge_index, num_nodes=len(coords))

                # 2. Матрица признаков X (Умная динамическая защита от разной длины векторов)
                x_features = []
                
                # Шаг А: Определяем целевую размерность эмбеддинга для текущего эксперта
                target_dim = None
                for c in registry.cells:
                    if self.expert_key in c.expert_embeddings:
                        emb = c.expert_embeddings[self.expert_key]
                        if emb is not None and hasattr(emb, '__len__') and len(emb) > 0:
                            current_len = len(emb)
                            # Берем максимальную длину (если есть смесь 10 и 100, выберем 100)
                            if target_dim is None or current_len > target_dim:
                                target_dim = current_len
                
                # Фоллбек, если эмбеддингов вообще не обнаружено
                if target_dim is None:
                    target_dim = 100

                # Шаг Б: Сборка фич с автоматическим выравниванием размерностей
                for c_idx, c in enumerate(registry.cells):
                    if self.expert_key not in c.expert_embeddings:
                        feat = [0.0] * target_dim
                    else:
                        feat = c.expert_embeddings[self.expert_key]
                        
                        # Безопасно приводим к стандартному python-list
                        if isinstance(feat, torch.Tensor):
                            feat = feat.detach().cpu().tolist()
                        elif isinstance(feat, np.ndarray):
                            feat = feat.tolist()
                        else:
                            feat = list(feat)
                        
                        # Если длина вектора клетки не совпадает с целевой — выравниваем её
                        if len(feat) != target_dim:
                            if len(feat) < target_dim:
                                # Дописываем нули (паддинг)
                                feat = feat + [0.0] * (target_dim - len(feat))
                            else:
                                # Обрезаем лишнее
                                feat = feat[:target_dim]
                                
                    x_features.append(feat)
                    
                x_tensor = torch.tensor(x_features, dtype=torch.float)

                # 3. Матрица таргетов Y (Защита от несовпадения list/dict)
                y_features = []
                for c_idx, c in enumerate(registry.cells):
                    cell_genes_data = c.segmentation_data.features["xenium_panel"].value
                    
                    if isinstance(cell_genes_data, dict):
                        gene_values = [float(cell_genes_data.get(gene, 0.0)) for gene in self.target_gene_names]
                    elif isinstance(cell_genes_data, list):
                        if not self.all_gene_names:
                            raise ValueError("Клетка хранит экспрессию списком, но список всех генов (all_gene_names) не передан в датасет!")
                        gene_values = []
                        for gene in self.target_gene_names:
                            g_idx = self.gene_to_idx.get(gene)
                            gene_values.append(float(cell_genes_data[g_idx]) if g_idx is not None else 0.0)
                    else:
                        raise TypeError(f"Неподдерживаемый тип экспрессии в ячейке: {type(cell_genes_data)}")
                        
                    y_features.append(gene_values)
                    
                y_tensor = torch.tensor(y_features, dtype=torch.float)

                graph_data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
                self._precomputed_graphs.append(graph_data)

            except Exception as e:
                print(f"❌ Ошибка сборки графа на регистре #{idx}: {str(e)}")

    def __len__(self): return len(self._precomputed_graphs)
    def __getitem__(self, idx): return self._precomputed_graphs[idx]
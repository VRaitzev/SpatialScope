import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from scipy.spatial import Delaunay

from .XeniumSpatialSegmentationAdapter import XeniumSpatialSegmentationAdapter

class XeniumSpatialGraphDataset(Dataset):
    def __init__(self, spatial_adapter: XeniumSpatialSegmentationAdapter, patch_locations: list, patch_size: tuple = (512, 512)):
        """
        Датасет, генерирующий пространственные графы клеток для PyTorch Geometric 
        на основе контейнера SpatialData.
        """
        self.adapter = spatial_adapter
        self.patch_locations = patch_locations
        self.patch_size = patch_size
        
        # Кэшируем метаданные слайда один раз при инициализации
        self.wsi_container = self.adapter.loader.load_slide_metadata("")

    def __len__(self):
        return len(self.patch_locations)

    def __getitem__(self, idx):
        x, y = self.patch_locations[idx]
        w, h = self.patch_size
        
        # 1. Загружаем регион изображения (картинка автоматически выравнивается под граф)
        patch_image = self.adapter.loader.load_region(
            self.wsi_container, x, y, w, h
        )
        
        # 2. Получаем контейнер клеток с их молекулярными профилями
        container = self.adapter.predict(patch_image)
        
        # 3. Строим граф из пространственного распределения клеток
        return self._build_graph(container)

    def _build_graph(self, container):
        num_genes = len(self.adapter.gene_loader.get_all_gene_names())
        
        # --- СЦЕНАРИЙ 1: В патче нет клеток (пустая область слайда) ---
        if not container.instances:
            return Data(
                x=torch.empty((0, num_genes), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.empty((0, num_genes), dtype=torch.float),
                coords=torch.empty((0, 2), dtype=torch.float)
            )

        # Извлекаем центроиды и векторы экспрессии
        coords = np.array([inst.centroid for inst in container.instances])
        features = np.array([inst.features["xenium_panel"].value for inst in container.instances])
        
        num_nodes = len(coords)
        
        # --- СЦЕНАРИЙ 2: Клеток слишком мало для триангуляции Делоне (< 4) ---
        # В этом случае строим полносвязный граф (каждая клетка связана с каждой)
        if num_nodes < 4:
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append((i, j))
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # --- СЦЕНАРИЙ 3: Стандартный граф Делоне ---
        else:
            try:
                tri = Delaunay(coords)
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i + 1, 3):
                            # Добавляем ребра в обе стороны (неориентированный граф)
                            edges.add((simplex[i], simplex[j]))
                            edges.add((simplex[j], simplex[i]))
                edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
                
            except Exception as e:
                # Фоллбэк на случай коллинеарности (клетки лежат на одной линии)
                print(f"⚠️ Геометрическая аномалия Делоне: {e}. Переключаемся на полносвязный граф.")
                edges = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            edges.append((i, j))
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Добавляем self-loops (петли), чтобы узел при агрегации GAT учитывал свои фичи
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        x_tensor = torch.tensor(features, dtype=torch.float)
        
        return Data(
            x=x_tensor,                                    # Фичи узлов (экспрессия генов)
            edge_index=edge_index,                         # Матрица связности микроокружения
            y=x_tensor,                                    # Целевая переменная
            coords=torch.tensor(coords, dtype=torch.float) # Координаты (пригодятся для контроля и XAI)
        )
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay
from torch_geometric.utils import add_self_loops
import numpy as np

class XeniumGraphDataset(Dataset):
    def __init__(self, xenium_adapter, patch_locations: list):
        self.adapter = xenium_adapter # Твой XeniumSegmentationAdapter
        self.patch_locations = patch_locations
        self.patch_size = (512, 512)

    def __len__(self):
        return len(self.patch_locations)

    def __getitem__(self, idx):
        x, y = self.patch_locations[idx]
        
        # 1. Используем твой адаптер для получения данных
        # Нам нужно сначала создать WSIRegionContainer (как в твоем пайплайне)
        # Для примера предполагаем, что патч уже загружен
        patch_image = self.adapter.loader.load_region(
            self.adapter.loader.load_slide_metadata("path_to_tif"), x, y, 512, 512
        )
        
        # 2. Получаем "предсказание" (инстансы клеток с генами)
        container = self.adapter.predict(patch_image)
        
        # 3. Строим граф из контейнера
        return self._build_graph(container)

    def _build_graph(self, container):
        # Извлекаем центроиды и векторы экспрессии
        coords = np.array([inst.centroid for inst in container.instances])
        # Берем значения генов из feat.value
        features = np.array([inst.features["xenium_panel"].value for inst in container.instances])
        
        # Строим топологию
        tri = Delaunay(coords)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add(tuple(sorted((simplex[i], simplex[j]))))
        
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(coords))
        
        return Data(
            x=torch.tensor(features, dtype=torch.float), # Здесь сейчас гены
            edge_index=edge_index,
            y=torch.tensor(features, dtype=torch.float) # Таргет = вход (автоэнкодер) или другое
        )
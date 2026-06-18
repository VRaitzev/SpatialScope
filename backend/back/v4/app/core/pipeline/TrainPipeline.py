import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from app.core.pipeline.AbstractTrainPipeline import AbstractTrainPipeline

class SpatialTrainPipeline(AbstractTrainPipeline):
    def __init__(
        self,
        dataset, 
        spatial_model,  # Сюда передаем наш SpatialGNNAdapter или саму TranscriptomicsGNN
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = nn.MSELoss(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.dataset = dataset
        self.device = device
        
        # Защита на случай, если передали адаптер вместо сырой модели PyTorch
        if hasattr(spatial_model, 'model'):
            self.spatial_model = spatial_model.model
        else:
            self.spatial_model = spatial_model
            
        self.spatial_model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def data_prepare(self, batch_size: int = 4, num_workers: int = 2) -> PyGDataLoader:
        """
        Используем нативный DataLoader из PyTorch Geometric.
        Он автоматически объединяет отдельные графы патчей в один большой 
        дизъюнктный граф (склеивает матрицы X и Y по вертикали, корректно смещая индексы ребер).
        """
        return PyGDataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )

    def train(self, epochs: int) -> None:
        """Оптимизированный цикл обучения графовой сети."""
        dataloader = self.data_prepare()

        for epoch in range(epochs):
            self.spatial_model.train()  # Переводим модель в режим обучения каждую эпоху
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for graph_batch in progress_bar:
                # Отправляем весь пакет графов на GPU за одну операцию
                graph_batch = graph_batch.to(self.device)
                
                self.optimizer.zero_grad()

                # 1. Прямой проход (Forward Pass)
                # Наша TranscriptomicsGNN принимает на вход тензор фич x и индексы ребер edge_index
                preds = self.spatial_model(graph_batch.x, graph_batch.edge_index)
                
                # 2. Извлечение таргетов
                # Благодаря PyG DataLoader, реальные значения экспрессии генов для всего батча 
                # уже склеены в правильном порядке внутри graph_batch.y
                ground_truth = graph_batch.y

                # 3. Расчет ошибки (Loss)
                loss = self.criterion(preds, ground_truth)
                
                # 4. Обратное распространение и шаг оптимизатора
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")
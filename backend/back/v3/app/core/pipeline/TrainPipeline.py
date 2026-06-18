import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from app.core.schemes.RNA import RegionalTargetOutput
from app.core.pipeline.AbstractTrainPipeline import AbstractTrainPipeline

class SpatialTrainPipeline(AbstractTrainPipeline):
    def __init__(
        self,
        dataset, 
        segmentation_model,
        expert_model,
        spatial_model,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = nn.MSELoss(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.dataset = dataset
        self.segmentation_model = segmentation_model.to(device).eval()
        self.expert_model = expert_model.to(device).eval()
        self.spatial_model = spatial_model.to(device).train()
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def data_prepare(self, batch_size: int = 4, num_workers: int = 2) -> DataLoader:
        """Даталоадер с использованием batching для графов."""
        return DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=self.dataset.collate_fn # Должна возвращать (List[Patch], List[Target], Batch)
        )

    def train(self, epochs: int) -> None:
        """Оптимизированный цикл обучения с поддержкой Batch-графов."""
        dataloader = self.data_prepare()

        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_data in progress_bar:
                # Предполагаем, что collate_fn возвращает подготовленный Batch объектов Data
                patches, targets, graph_batch = batch_data
                graph_batch = graph_batch.to(self.device)
                
                self.optimizer.zero_grad()

                # 1. Получаем предсказания от GNN (один проход для всего батча!)
                # Модель должна уметь работать с graph_batch (x, edge_index, batch_vector)
                preds = self.spatial_model(graph_batch)
                
                # 2. Подготовка таргетов
                # Нужно собрать все векторы экспрессии из targets в один тензор
                ground_truth = torch.cat([
                    torch.tensor(t.expression_matrix, dtype=torch.float32) 
                    for t in targets
                ]).to(self.device)

                # 3. Лосс
                loss = self.criterion(preds, ground_truth)
                
                # 4. Обратное распространение
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss / len(dataloader):.4f}")
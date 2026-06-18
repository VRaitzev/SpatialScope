import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
import os

from app.core.pipeline.AbstractTrainPipeline import AbstractTrainPipeline

class SpatialTrainPipeline(AbstractTrainPipeline):
    def __init__(
        self,
        dataset, 
        spatial_model,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = nn.MSELoss(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints",  # <- НОВЫЙ ПАРАМЕТР
        model_name: str = "best_model"    # <- НОВЫЙ ПАРАМЕТР
    ):
        self.dataset = dataset
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        
        # Создаём папку для чекпоинтов
        os.makedirs(save_dir, exist_ok=True)
        
        # Защита на случай, если передали адаптер вместо сырой модели PyTorch
        if hasattr(spatial_model, 'model'):
            self.spatial_model = spatial_model.model
        else:
            self.spatial_model = spatial_model
            
        self.spatial_model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Для отслеживания лучшей модели
        self.best_loss = float('inf')

    def data_prepare(self, batch_size: int = 4, num_workers: int = 2) -> PyGDataLoader:
        """
        Используем нативный DataLoader из PyTorch Geometric.
        """
        # ИСПРАВЛЕНО: проверка на пустой датасет
        if len(self.dataset) == 0:
            raise ValueError("Датасет пуст! Невозможно создать DataLoader.")
        
        return PyGDataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Сохраняет чекпоинт модели."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.spatial_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        # Всегда сохраняем последний
        torch.save(checkpoint, os.path.join(self.save_dir, f"{self.model_name}_last.pt"))
        
        # Если это лучшая модель — сохраняем отдельно
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, f"{self.model_name}_best.pt"))
            print(f"🏆 [Train] Сохранена лучшая модель с loss: {loss:.4f}")

    def train(self, epochs: int, save_every: int = 5) -> None:
        """
        Оптимизированный цикл обучения графовой сети.
        
        Args:
            epochs: Количество эпох обучения
            save_every: Сохранять чекпоинт каждые N эпох
        """
        # ИСПРАВЛЕНО: проверка на пустой датасет
        if len(self.dataset) == 0:
            raise ValueError("Датасет пуст! Обучение невозможно.")
        
        dataloader = self.data_prepare()
        
        # ИСПРАВЛЕНО: если в датасете только 1 граф, batch_size=1
        if len(self.dataset) == 1 and dataloader.batch_size > 1:
            print("⚠️ [Train] Датасет содержит только 1 граф, уменьшаю batch_size до 1")
            dataloader = self.data_prepare(batch_size=1)

        for epoch in range(epochs):
            self.spatial_model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for graph_batch in progress_bar:
                graph_batch = graph_batch.to(self.device)
                
                self.optimizer.zero_grad()

                # Прямой проход
                preds = self.spatial_model(graph_batch.x, graph_batch.edge_index)
                
                # Проверка размерностей
                if preds.shape != graph_batch.y.shape:
                    print(f"⚠️ [Train] Несовпадение размерностей: preds {preds.shape} vs y {graph_batch.y.shape}")
                    # Обрезаем или дополняем
                    min_dim = min(preds.shape[1], graph_batch.y.shape[1])
                    preds = preds[:, :min_dim]
                    ground_truth = graph_batch.y[:, :min_dim]
                else:
                    ground_truth = graph_batch.y

                # Расчет ошибки
                loss = self.criterion(preds, ground_truth)
                
                # Обратное распространение
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_epoch_loss = epoch_loss / len(dataloader)
            
            # ИСПРАВЛЕНО: сохраняем лучшую модель
            is_best = avg_epoch_loss < self.best_loss
            if is_best:
                self.best_loss = avg_epoch_loss
            
            # Сохраняем чекпоинт
            if (epoch + 1) % save_every == 0 or is_best:
                self._save_checkpoint(epoch + 1, avg_epoch_loss, is_best)
            
            print(f"Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_epoch_loss:.4f} {'⭐ BEST' if is_best else ''}")

        # ИСПРАВЛЕНО: переводим модель в режим оценки после обучения
        self.spatial_model.eval()
        print(f"✅ [Train] Обучение завершено. Лучший loss: {self.best_loss:.4f}")
        print(f"   💾 Чекпоинты сохранены в: {self.save_dir}")

    def load_best_model(self):
        """Загружает лучшую сохранённую модель."""
        best_path = os.path.join(self.save_dir, f"{self.model_name}_best.pt")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.spatial_model.load_state_dict(checkpoint['model_state_dict'])
            self.spatial_model.eval()
            print(f"✅ [Train] Загружена лучшая модель с loss: {checkpoint['loss']:.4f}")
            return checkpoint
        else:
            print(f"⚠️ [Train] Файл {best_path} не найден")
            return None
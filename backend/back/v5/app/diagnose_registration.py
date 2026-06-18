import sys
from pathlib import Path
import numpy as np
import pandas as pd
import ants
import matplotlib.pyplot as plt
import matplotlib.patches as patches

root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

# Импортируем обновленные компоненты
from app.core.loaders.wsi.image.XeniumWSILoader import XeniumWSILoader
from app.core.loaders.rna.XeniumGenePanelLoader import XeniumGenePanelLoader
from app.core.loaders.wsi.segmentatioin.XeniumCellBoundaryAdapter import XeniumCellBoundaryAdapter
from app.core.ml_models.wsi_models.registration_models.SpatialRegistrationService import SpatialRegistrationService

# 1. Абсолютные пути к данным
data_dir = Path(r"D:\PROJECTS\Personal\Study\workspace\diploma\data\full_dataset")
cells_path = str(data_dir / "cells.parquet")
matrix_path = str(data_dir / "cell_feature_matrix.h5")

# Файлы изображений для нелинейной регистрации
dapi_image_path = str(data_dir / "morphology.ome.tif")
he_image_path = str(data_dir / "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_he_image.ome.tif")
cache_directory = str(data_dir / "registration_cache")

# 2. Инициализация сервиса пространственной деформации
reg_service = SpatialRegistrationService(
    dapi_path=dapi_image_path,
    he_path=he_image_path,
    cache_dir=cache_directory,
    target_max_dim=4000,   # Сжатие для ускорения расчетов
    mpp_xenium=0.2125
)

# ФИКС: Теперь метод возвращает 3 значения (картинку, шаг даунсэмпла и смещение кропа)
dapi_img, dapi_ds, dapi_offset = reg_service._load_downsampled_gray(dapi_image_path, 4000, is_he=False)
he_img, he_ds, he_offset = reg_service._load_downsampled_gray(he_image_path, 4000, is_he=True)

print(f"🟢 DAPI: Матрица ткани {dapi_img.shape}, Downsample={dapi_ds:.4f}, Offset={dapi_offset}")
print(f"🟢 HE:   Матрица ткани {he_img.shape}, Downsample={he_ds:.4f}, Offset={he_offset}")

fixed = ants.from_numpy(he_img.astype("float32"))
moving = ants.from_numpy(dapi_img.astype("float32"))

# Применяем прямую трансформацию картинок (уже внутри вырезанного bbox ткани)
warped_dapi = ants.apply_transforms(
    fixed=fixed, 
    moving=moving,
    transformlist=reg_service.reg["fwdtransforms"]
)

# Визуализация оверлея
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(he_img, cmap="gray")
axes[0].set_title("H&E (fixed, cropped)")

axes[1].imshow(warped_dapi.numpy(), cmap="gray")
axes[1].set_title("Warped DAPI (moving -> HE space)")

# Накладываем каналы друг на друга: H&E в красный, DAPI в зеленый
axes[2].imshow(he_img, cmap="Reds", alpha=0.5)
axes[2].imshow(warped_dapi.numpy(), cmap="Greens", alpha=0.5)
axes[2].set_title("Overlay Check (Focused)")

plt.tight_layout()
plt.show()
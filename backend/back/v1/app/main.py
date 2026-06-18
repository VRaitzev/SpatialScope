import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tifffile
# 1. Настройка путей для импортов
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from app.core.loaders.wsi.image.XeniumWSILoader import XeniumWSILoader
from app.core.loaders.rna.XeniumGenePanelLoader import XeniumGenePanelLoader
from app.core.ml_models.wsi_models.segmentations_models.XeniumSegmentationAdapter import XeniumSegmentationAdapter


def print_ome_metadata(image_path):
    with tifffile.TiffFile(image_path) as tif:
        # Выводим описание OME-XML
        print("📋 Метаданные OME-XML:")
        print(tif.ome_metadata) 
        
        # Выводим информацию о каналах (если она есть)
        if 'Channel' in tif.ome_metadata:
            for channel in tif.ome_metadata['Channel']:
                print(f"Канал {channel.get('ID')}: {channel.get('Name')}")
        
        # Показываем структуру пирамиды (уровни разрешения)
        print("\n📈 Уровни пирамиды:")
        for i, page in enumerate(tif.series[0].pages):
                print(f"Уровень {i}: {page.shape} (Downsample: {2**i}x)")

def demo_real_xenium():
    # --- НАСТРОЙКИ ---
    ZOOM_FACTOR = 1.0
    BASE_SIZE = 1024
    
    data_dir = Path(r"D:\PROJECTS\Personal\Study\Новая папка\diploma\data\full_dataset")
    cells_path = str(data_dir / "cells.parquet")
    image_path = str(data_dir / "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_he_image.ome.tif")
    matrix_path = str(data_dir / "cell_feature_matrix.h5")
    
    # Инициализация
    loader = XeniumWSILoader(cells_path=cells_path)
    gene_loader = XeniumGenePanelLoader(matrix_path=matrix_path)
    adapter = XeniumSegmentationAdapter(loader, gene_loader)
    container = loader.load_slide_metadata(image_path)
    
    # Координаты
    w, h = container.dimensions_level_0
    x_start = int(w * 0.4)
    y_start = int(h * 0.4)
    
    print(f"🖼 Загрузка региона: {x_start}, {y_start}")
    region = loader.load_region(container, x=x_start, y=y_start, width=BASE_SIZE, height=BASE_SIZE)
    seg_container = adapter.predict(region)
    
    # --- УМНАЯ ВИЗУАЛИЗАЦИЯ ---
    # Переводим обратно из (C, H, W) в (H, W, C) для matplotlib
    patch_viz = region.image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(patch_viz) # Теперь здесь гарантированно будет цветной H&E!
    
    # Отрисовка полигонов
    for cell in seg_container.instances:
        if cell.polygon and cell.polygon.points:
            poly_points = [(p[0] - region.x_wsi, p[1] - region.y_wsi) for p in cell.polygon.points]
            polygon = patches.Polygon(poly_points, closed=True, edgecolor='lime', facecolor='none', linewidth=0.8, alpha=0.9)
            ax.add_patch(polygon)

    centers_x = [cell.centroid[0] - region.x_wsi for cell in seg_container.instances if cell.centroid]
    centers_y = [cell.centroid[1] - region.y_wsi for cell in seg_container.instances if cell.centroid]
    
    ax.scatter(centers_x, centers_y, color='red', s=0.5, alpha=0.5, label='Centroids')
    plt.legend(loc='upper right')
    
    plt.title(f"Xenium Analysis | Cells: {len(seg_container.instances)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # print_ome_metadata(r"D:\PROJECTS\Personal\Study\Новая папка\diploma\data\full_dataset\Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_he_image.ome.tif")
    demo_real_xenium()
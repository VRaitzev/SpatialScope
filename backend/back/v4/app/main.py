import os
import torch
import random
import numpy as np
from tqdm import tqdm

# 1. Импорт базовых контейнеров из core-системы
from app.core.schemes.WSI import WSIRegionContainer

# 2. Импорт компонентов загрузки и обработки данных Xenium
from app.core.loaders.wsi.image.XeniumWSILoader import XeniumWSILoader
from app.core.loaders.wsi.segmentatioin.XeniumCellBoundaryAdapter import XeniumCellBoundaryAdapter
from app.core.loaders.rna.XeniumGenePanelLoader import XeniumGenePanelLoader
from app.core.loaders.connector.XeniumGraphDataset import XeniumGraphDataset

# 3. Импорт экспертных адаптеров
from app.core.ml_models.wsi_models.expert_models.RadiomicsExpertAdapter import RadiomicsExpertAdapter
from app.core.ml_models.wsi_models.expert_models.EfficientNetExpertAdapter import EfficientNetExpertAdapter

# 4. Компоненты пространственного моделирования и пайплайна
from app.core.ml_models.wsi_models.spatial_model.SpatialGNNAdapter import SpatialGNNAdapter
from app.core.pipeline.TrainPipeline import SpatialTrainPipeline


def main():
    # ===============================================================
    # ⚙️ НАСТРОЙКА РЕЖИМА: Укажи "radiomics" или "cnn"
    # ===============================================================
    EXPERT_MODE = "radiomics"  
    # ===============================================================

    # --- НАСТРОЙКА ПУТЕЙ И ПАРАМЕТРОВ ОПТИМИЗАЦИИ ---
    DATA_DIR = "./data/Breast_Aligned_v102_outs" 
    
    CELLS_PARQUET = os.path.join(DATA_DIR, "cells.parquet")
    MATRIX_HDF5 = os.path.join(DATA_DIR, "cell_feature_matrix.h5")
    
    # Имя файла весов зависит от выбранной модели
    WEIGHTS_PATH = os.path.join(DATA_DIR, f"spatial_gat_breast_{EXPERT_MODE}_weights.pth")
    HE_IMAGE_TIF = os.path.join(DATA_DIR, "Xenium_V1_FFPE_Human_Breast_IDC_he_image.ome.tif")
    
    TARGET_GENES = ["ACTA2", "KRT8", "KRT5", "MKI67", "ERBB2", "CD3D", "PECAM1"]
    
    # [БЫСТРЫЙ ДЕБАГ-ПРОФИЛЬ]
    MIN_CELLS = 200          
    MAX_TRAIN_PATCHES = 2    
    MAX_VAL_PATCHES = 1      
    TOTAL_REQUIRED = MAX_TRAIN_PATCHES + MAX_VAL_PATCHES
    DEBUG_EPOCHS = 2         

    print("\n" + "="*60)
    print(f"🚀 ЗАПУЩЕН ПАЙПЛАЙН С ЭКСПЕРТОМ: {EXPERT_MODE.upper()}")
    print("="*60)
    
    print("\nШаг 1: Инициализация компонентов данных...")
    wsi_loader = XeniumWSILoader(cells_path=CELLS_PARQUET)
    gene_loader = XeniumGenePanelLoader(matrix_path=MATRIX_HDF5)
    boundary_adapter = XeniumCellBoundaryAdapter(wsi_loader, gene_loader)
    
    # Динамически создаем нужного эксперта и берем его уникальный ключ
    if EXPERT_MODE == "radiomics":
        expert = RadiomicsExpertAdapter()
        expert_key = RadiomicsExpertAdapter.EXPERT_KEY
    elif EXPERT_MODE == "cnn":
        expert = EfficientNetExpertAdapter()
        expert_key = EfficientNetExpertAdapter.EXPERT_KEY
    else:
        raise ValueError(f"Неизвестный режим эксперта: {EXPERT_MODE}. Используй 'radiomics' или 'cnn'.")
        
    all_gene_names = gene_loader.get_all_gene_names()
    slide_container = wsi_loader.load_slide_metadata(HE_IMAGE_TIF)
    w_wsi, h_wsi = slide_container.dimensions_level_0
    print(f"✅ Слайд загружен. Размерность слайда: {w_wsi}x{h_wsi}")

    print(f"\n✂️ Шаг 2: Сбор патчей и извлечение признаков клеток...")
    patch_size, stride = 4000, 4000  
    collected_registries = []
    
    # Жестко фиксируем координаты патчей, чтобы мгновенно подхватывать их из кэша
    all_positions = [
        (36000, 44000), 
        (8000, 4000), 
        (28000, 32000)
    ]
    print(f"📌 СМОК-ТЕСТ: Зафиксировано {len(all_positions)} готовых патчей для моментальной загрузки.")
    
    patch_idx = 0
    for x, y in all_positions:
        if len(collected_registries) >= TOTAL_REQUIRED:
            print(f"\n🎯 Собрано достаточно патчей ({TOTAL_REQUIRED}). Прерываем поиск.")
            break
            
        # 1. Фильтр по плотности клеток
        passports_check = wsi_loader.get_cells_in_region(x, y, patch_size, patch_size, use_he=False)
        if len(passports_check) < MIN_CELLS:
            print(f"    ⚠️ Пропуск патча X={x}, Y={y}: слишком мало клеток ({len(passports_check)} < {MIN_CELLS})")
            continue
        
        patch_idx += 1
        print(f"\n--- [ПАТЧ #{patch_idx}] Координаты: X={x}, Y={y} ---")
        
        # 2. Загрузка геометрии и изображений
        patch_region = wsi_loader.load_region(slide_container, x, y, patch_size, patch_size)
        segmentation_container = boundary_adapter.load_boundaries(patch_region, registration_service=None)
        
        num_instances = len(segmentation_container.instances) if segmentation_container.instances else 0
        if num_instances == 0:
            print("    ⚠️ Пропуск: нет сегментированных клеток.")
            continue
            
        # 3. Извлечение признаков через выбранного эксперта
        print(f"    [🔍] Запуск {expert.__class__.__name__}...")
        unique_id = f"patch_X{x}_Y{y}"
        patch_cell_registry = expert.predict(
            patch_region.image_tensor, 
            segmentation_container, 
            patch_id=unique_id
        )
        
        if patch_cell_registry is None:
            print(f"    ❌ Ошибка: {expert.__class__.__name__}.predict вернул None!")
            continue
            
        cells_in_registry = len(patch_cell_registry.cells) if patch_cell_registry.cells else 0
        if cells_in_registry == 0:
            print("    ⚠️ Пропуск: эксперт не вернул дескрипторы для клеток.")
            continue
            
        # Подготовка векторов экспрессии генов
        converted_count = 0
        for cell in patch_cell_registry.cells:
            if "xenium_panel" not in cell.segmentation_data.features:
                continue
                
            raw_list_expr = cell.segmentation_data.features["xenium_panel"].value
            if isinstance(raw_list_expr, (list, np.ndarray)):
                gene_dict = {name: raw_list_expr[idx] for idx, name in enumerate(all_gene_names)}
                cell.segmentation_data.features["xenium_panel"].value = gene_dict
                cell.segmentation_data.features["xenium_panel"].feature_name = "xenium_panel"
                converted_count += 1
                
        print(f"    ✅ Обработано клеток: {cells_in_registry} | Профилей генов синхронизировано: {converted_count}")
        collected_registries.append(patch_cell_registry)

    if len(collected_registries) == 0:
        print("\n❌ ОСТАНОВКА: Нет валидных патчей для сборки графов.")
        return
        
    train_registries = collected_registries[:MAX_TRAIN_PATCHES]
    val_registries = collected_registries[MAX_TRAIN_PATCHES:]

    print("\n📦 Шаг 3: Создание PyTorch Geometric Графовых Датасетов...")
    train_dataset = XeniumGraphDataset(
        registries=train_registries, 
        target_gene_names=TARGET_GENES,
        all_gene_names=all_gene_names,
        expert_key=expert_key
    )
    
    val_dataset = XeniumGraphDataset(
        registries=val_registries, 
        target_gene_names=TARGET_GENES,
        all_gene_names=all_gene_names,
        expert_key=expert_key
    )
    
    print(f"📊 Датасеты собраны. Train подграфов: {len(train_dataset)}, Val подграфов: {len(val_dataset)}")

    print("\n🏗 Шаг 4: Построение архитектуры Spatial GNN Network...")
    computed_feature_dim = train_dataset[0].x.shape[1]
    print(f"📐 Размерность входного вектора для GNN (определено автоматически): {computed_feature_dim}")
    
    model_adapter = SpatialGNNAdapter(
        feature_dim=computed_feature_dim, 
        hidden_dim=64, 
        gene_names=TARGET_GENES, 
        expert_key=expert_key,
        spot_radius=50.0
    )
    optimizer = torch.optim.Adam(model_adapter.model.parameters(), lr=0.001, weight_decay=1e-4)
    
    pipeline = SpatialTrainPipeline(
        dataset=train_dataset, 
        spatial_model=model_adapter,
        optimizer=optimizer, 
        criterion=torch.nn.MSELoss()
    )

    print("\n🏋️‍♂️ Шаг 5: Запуск цикла обучения (Дебаг)...")
    pipeline.train(epochs=DEBUG_EPOCHS)
    
    print("\n💾 Шаг 6: Сохранение результатов...")
    model_adapter.save_weights(WEIGHTS_PATH)
    print(f"🎉 Всё готово! Обучение завершено, веса сохранены тут: {WEIGHTS_PATH}\n")


if __name__ == "__main__":
    os.makedirs("./data/Breast_Aligned_v102_outs", exist_ok=True)
    main()
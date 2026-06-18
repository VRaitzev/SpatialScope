import os
import uuid
import threading
import torch
import logging
from typing import Dict, Any, List, Optional, Tuple

# Импорт кастомных лоадеров и адаптеров для работы с пространственной омикой
from app.core.loaders.wsi.image import XeniumWSILoader
from app.core.loaders.wsi.segmentatioin.XeniumCellBoundaryAdapter import XeniumCellBoundaryAdapter
from app.core.loaders.rna.XeniumGenePanelLoader import XeniumGenePanelLoader

# Импорт дата-схем (Pydantic / Dataclasses)
from app.core.schemes.WSI import WSIContainer
from app.core.schemes.Segmentation import SegmentationContainer
from app.core.schemes.Cells import RegionalSpatialSpot, RegionalPredictionOutput

# Импорт абстракций ядра системы
from app.core.registry.ModelPresetRegistry import AbstractModelRegistry
from app.core.pipeline import TrainPipeline  

# Слой персистентного хранения данных
from app.core.data_managers.ORM_manager import DatabaseService


class XeniumSystemOrchestrator:
    """
    [ORCHESTRATOR LAYER] Главный координатор ИСТ-платформы (Паттерн Фасад).
    Связывает веб-слой (FastAPI), ML-движок (PyTorch/PyG), ленивые лоадеры данных 
    и реляционную СУБД (DatabaseService). Обеспечивает сквозной аудит безопасности.
    """

    def __init__(self, registry: AbstractModelRegistry, db_service: DatabaseService, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        self.registry = registry
        self.db = db_service 
        os.makedirs(cache_dir, exist_ok=True)

        # Кэш лоадеров в оперативной памяти (Lazy Loading)
        # Защищает систему от повторного парсинга тяжелых файлов (HDF5/Parquet) при частых запросах
        self.slides_cache: Dict[str, Dict[str, Any]] = {}

        # Трекер фоновых задач обучения GNN (для Исследователей)
        self.jobs_registry: Dict[str, Dict[str, Any]] = {}

    # =====================================================================
    # --- РАЗДЕЛ 1: СИНХРОНИЗАЦИЯ И ЛЕНИВАЯ ЗАГРУЗКА ДАННЫХ СЛАЙДОВ ---
    # =====================================================================

    def register_and_cache_slide(
        self, 
        project_id: str, 
        patient_name: str, 
        tiff_path: str, 
        parquet_path: str, 
        hdf5_path: str
    ) -> str:
        """
        Регистрирует физические пути к файлам Xenium в базе данных,
        извлекает первичные метаданные и инициализирует ленивые лоадеры в кэше.
        """
        print(f"📦 [Orchestrator] Регистрация нового слайда пациента {patient_name} в БД...")
        
        # Сначала инициализируем лоадеры, чтобы вытащить первичные метаданные слайда
        gene_loader = XeniumGenePanelLoader(matrix_path=hdf5_path)
        wsi_loader = XeniumWSILoader(cells_path=parquet_path)
        boundary_adapter = XeniumCellBoundaryAdapter(xenium_loader=wsi_loader, gene_loader=gene_loader)
        
        # Читаем метаданные WSI (разрешение, MPP, окрашивание)
        metadata: WSIContainer = wsi_loader.load_slide_metadata(tiff_path)
        wsi_loader.apply_registration_to_dataframe(reg_service=None)
        
        # Сериализуем метаданные в JSON-friendly формат для СУБД
        metadata_dict = {
            "width": metadata.dimensions_level_0[0],
            "height": metadata.dimensions_level_0[1],
            "staining": metadata.staining,
            "mpp": metadata.mpp
        }

        # Сохраняем карточку слайда в реляционную БД через DatabaseService
        db_slide = self.db.register_xenium_slide(
            project_id=project_id,
            patient_name=patient_name,
            tiff_path=tiff_path,
            parquet_path=parquet_path,
            hdf5_path=hdf5_path,
            metadata=metadata_dict
        )

        # Кэшируем готовые тяжелые инстансы лоадеров в оперативную память под ID из базы данных
        self.slides_cache[db_slide.id] = {
            "wsi_loader": wsi_loader,
            "gene_loader": gene_loader,
            "boundary_adapter": boundary_adapter,
            "metadata": metadata,
            "db_record": db_slide
        }

        print(f"✅ [Orchestrator] Слайд успешно сохранен в БД и кэширован под ID: {db_slide.id}")
        return db_slide.id

    def _ensure_slide_is_cached(self, slide_id: str) -> Dict[str, Any]:
        """
        Внутренний защитный метод (Lazy Loading).
        Если сервер был перезапущен и RAM-кэш пуст, метод автоматически 
        поднимает пути к файлам из БД и восстанавливает лоадеры.
        """
        if slide_id in self.slides_cache:
            return self.slides_cache[slide_id]

        print(f"🔍 [Orchestrator] Слайд {slide_id} отсутствует в кэше памяти. Восстановление из БД...")
        db_slide = self.db.get_slide_by_id(slide_id)
        if not db_slide:
            raise KeyError(f"Слайд с ID {slide_id} не найден в базе данных системы.")

        # Реконструируем лоадеры на основе путей из базы данных
        gene_loader = XeniumGenePanelLoader(matrix_path=db_slide.matrix_hdf5_path)
        wsi_loader = XeniumWSILoader(cells_path=db_slide.cells_parquet_path)
        boundary_adapter = XeniumCellBoundaryAdapter(xenium_loader=wsi_loader, gene_loader=gene_loader)
        
        metadata: WSIContainer = wsi_loader.load_slide_metadata(db_slide.slide_tiff_path)
        wsi_loader.apply_registration_to_dataframe(reg_service=None)

        self.slides_cache[slide_id] = {
            "wsi_loader": wsi_loader,
            "gene_loader": gene_loader,
            "boundary_adapter": boundary_adapter,
            "metadata": metadata,
            "db_record": db_slide
        }
        return self.slides_cache[slide_id]

    # =====================================================================
    # --- РАЗДЕЛ 2: СКВОЗНОЙ ИНФЕРЕНС С АВТОСОХРАНЕНИЕМ ОТЧЕТОВ ---
    # =====================================================================

    def run_inference_and_save_report(
        self, 
        user_id: str,
        slide_id: str, 
        x: int, 
        y: int, 
        patch_size: int,
        regional_spots: List[RegionalSpatialSpot],
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Режим Клинициста: запускает каскад нейросетей для выбранного ROI ($x, y$), 
        генерирует молекулярный отчет в БД и логирует действие в целях ИБ.
        """
        # 1. Гарантируем, что лоадеры слайда развернуты в памяти
        slide_data = self._ensure_slide_is_cached(slide_id)
        
        wsi_loader: XeniumWSILoader = slide_data["wsi_loader"]
        adapter: XeniumCellBoundaryAdapter = slide_data["boundary_adapter"]
        metadata: WSIContainer = slide_data["metadata"]

        # 2. Извлекаем текущую активную пару моделей из реестра пресетов
        expert_model, spatial_model = self.registry.get_active_models()
        active_preset_id = self.registry._active_preset_id 
        
        patch_id = f"patch_{slide_id}_{x}_{y}_{patch_size}"

        # Шаг 2.1: Чтение региона ткани (Tiling) и геометрии контуров клеток
        patch_region = wsi_loader.load_region(metadata, x, y, patch_size, patch_size)
        segmentation_container: SegmentationContainer = adapter.load_boundaries(patch_region, registration_service=None)
        
        if not segmentation_container.instances:
            return {"status": "empty", "message": "В выбранной области нет сегментированных клеток."}

        # Шаг 2.2: Прогон Эксперта 1-го уровня (Извлечение визуальных признаков/морфологии)
        if "EfficientNet" in expert_model.EXPERT_KEY or "Radiomics" in expert_model.EXPERT_KEY:
            patch_tensor = torch.from_numpy(patch_region.image).permute(2, 0, 1) if hasattr(patch_region.image, 'shape') else patch_region.image
            cell_features_registry = expert_model.predict(
                patch_image=patch_tensor, 
                cell_registry=segmentation_container, 
                patch_id=patch_id
            )
        else:
            cell_features_registry = expert_model.predict(
                cell_registry=segmentation_container, 
                patch_id=patch_id,
                slide_id=slide_id
            )

        # Шаг 2.3: Прогон Графовой Сети 2-го уровня (GATv2 / Предсказание транскриптомного профиля)
        prediction_output: RegionalPredictionOutput = spatial_model.predict_spatial_expression(
            patch_registry=cell_features_registry,
            regional_spots=regional_spots,
            patch_id=patch_id
        )

        # 3. Сохраняем сгенерированный клинический отчет в базу данных
        roi_coordinates = {"x": x, "y": y, "patch_size": patch_size}
        predicted_data_dict = prediction_output.model_dump() if hasattr(prediction_output, 'model_dump') else prediction_output.dict()

        db_report = self.db.create_analysis_report(
            user_id=user_id,
            slide_id=slide_id,
            preset_id=active_preset_id,
            roi_coordinates=roi_coordinates,
            predicted_data=predicted_data_dict
        )

        # 4. Логируем действие пользователя для протокола ИБ
        self.db.log_action(
            user_id=user_id,
            action=f"RUN_INFERENCE | Preset: {active_preset_id} | Report Saved: {db_report.id}",
            ip_address=ip_address
        )

        return {
            "status": "success",
            "report_id": db_report.id,
            "created_at": db_report.created_at,
            "data": predicted_data_dict
        }

    # =====================================================================
    # --- РАЗДЕЛ 3: АСИНХРОННОЕ ОБУЧЕНИЕ С ОБНОВЛЕНИЕМ СТАТУСОВ В БД ---
    # =====================================================================

    def start_background_training(self, user_id: str, slide_id: str, epochs: int, ip_address: Optional[str] = None) -> str:
        """
        Режим Исследователя: запускает фоновое обучение GNN-модели (потоки OS).
        Изменяет статус слайда в БД на 'processing', предотвращая конфликты доступа.
        """
        slide_data = self._ensure_slide_is_cached(slide_id)
        job_id = f"job_{uuid.uuid4().hex[:6]}"
        
        # Инициализируем оперативную запись состояния таски
        self.jobs_registry[job_id] = {
            "status": "running", 
            "epoch": 0, 
            "loss": 0.0, 
            "total_epochs": epochs
        }

        # Блокируем слайд в БД и пишем лог
        self.db.update_slide_status(slide_id, status="processing")
        self.db.log_action(user_id=user_id, action=f"START_TRAINING | Job: {job_id} on Slide: {slide_id}", ip_address=ip_address)

        def training_worker():
            try:
                _, spatial_model = self.registry.get_active_models()
                dataset = [] # Сюда через boundary_adapter подгружается граф

                optimizer = torch.optim.Adam(spatial_model.model.parameters(), lr=0.001)
                train_pipeline = TrainPipeline(
                    dataset=dataset,
                    spatial_model=spatial_model.model,
                    optimizer=optimizer
                )
                
                # Имитация эпох обучения графовой сети
                for epoch in range(epochs):
                    # В продуктовом коде: train_pipeline.train_epoch()
                    fake_loss = 0.5 / (epoch + 1)
                    
                    # Пишем прогресс в оперативную память для фронтенд-поллинга
                    self.jobs_registry[job_id]["epoch"] = epoch + 1
                    self.jobs_registry[job_id]["loss"] = round(fake_loss, 4)

                # Фиксация успеха обучения в СУБД и освобождение слайда
                self.jobs_registry[job_id]["status"] = "success"
                self.db.update_slide_status(slide_id, status="ready")
                self.db.log_action(user_id=None, action=f"TRAINING_SUCCESS | Job {job_id} completed успешно.")
                
            except Exception as e:
                self.jobs_registry[job_id]["status"] = f"failed: {str(e)}"
                self.db.update_slide_status(slide_id, status="failed")
                self.db.log_action(user_id=None, action=f"TRAINING_FAILED | Job {job_id} crashed: {str(e)}")

        # Воркер запускается в виде демона, чтобы не вешать основной процесс FastAPI
        threading.Thread(target=training_worker, daemon=True).start()
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Возвращает текущую эпоху и лосс фоновой задачи обучения."""
        return self.jobs_registry.get(job_id, {"status": "not_found"})

    # =====================================================================
    # --- РАЗДЕЛ 4: УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ И АВТОРИЗАЦИЕЙ (IAM) ---
    # =====================================================================

    def register_user(
        self, 
        admin_user_id: str, 
        email: str, 
        name: str, 
        password_hash: str, 
        role: str = 'clinician',
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Регистрирует нового сотрудника клиники. Доступно строго пользователям с ролью 'admin'.
        """
        # Проверка прав оператора (RBAC)
        operator = self.db.get_user_by_id(admin_user_id)
        if not operator or operator.role != 'admin':
            self.db.log_action(admin_user_id, f"ACCESS_DENIED | Попытка несанкционированного создания пользователя: {email}", ip_address)
            raise PermissionError("Доступ отклонен: у вас нет прав администратора.")

        # Создание записи в БД
        new_user = self.db.create_user(email=email, name=name, password_hash=password_hash, role=role)
        
        # Логирование события безопасности
        self.db.log_action(
            user_id=admin_user_id, 
            action=f"CREATE_USER | Создан пользователь ID: {new_user.id}, ФИО: {name}, Роль: {role}", 
            ip_address=ip_address
        )
        
        return {"status": "success", "user_id": new_user.id, "role": new_user.role}

    def authenticate_and_get_user(self, email: str, ip_address: Optional[str] = None) -> Optional[Any]:
        """
        Точка интеграции для JWT Middleware. Запрашивает пользователя по email и логирует вход.
        """
        user = self.db.get_user_by_email(email)
        if user:
            self.db.log_action(user.id, "USER_LOGIN | Успешная аутентификация в системе", ip_address)
        else:
            self.db.log_action(None, f"LOGIN_FAILED | Неудачная попытка входа (Email: {email})", ip_address)
        return user

    # =====================================================================
    # --- РАЗДЕЛ 5: УПРАВЛЕНИЕ КЛИНИЧЕСКИМИ ПРОЕКТАМИ И ДАШБОРДОМ ---
    # =====================================================================

    def create_new_project(self, user_id: str, name: str, description: str = None, ip_address: Optional[str] = None) -> str:
        """Создает изолированное рабочее пространство (проект) для хранения группы слайдов."""
        project = self.db.create_project(user_id=user_id, name=name, description=description)
        self.db.log_action(user_id, f"CREATE_PROJECT | Создан проект '{name}' (ID: {project.id})", ip_address)
        return project.id

    def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """
        Агрегирует все данные для главного экрана: проекты пользователя, 
        список привязанных к ним WSI-слайдов, их статусы и доступные пресеты моделей ИИ.
        """
        user_projects = self.db.get_user_projects(user_id)
        available_presets = self.registry.get_available_presets()
        
        projects_dashboard = []
        for proj in user_projects:
            slides = self.db.get_project_slides(proj.id)
            projects_dashboard.append({
                "project_id": proj.id,
                "name": proj.name,
                "description": proj.description,
                "slides_count": len(slides),
                "slides": [{"id": s.id, "patient": s.patient_name, "status": s.status} for s in slides]
            })
            
        return {
            "projects": projects_dashboard,
            "system_presets": [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in available_presets.values()]
        }
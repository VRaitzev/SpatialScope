import os
import logging
from typing import Optional, List
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Импорт нашей системы
from app.core.orchestrators.XeniumSystemOrchestrator import XeniumSystemOrchestrator
from app.core.registry.ModelPresetRegistry import ModelPresetRegistry
from app.core.data_managers.ORM_manager import DatabaseService
from app.core.schemes.Cells import RegionalSpatialSpot
from app.core.ml_models.wsi_models.expert_models import (
    EfficientNetExpertAdapter,
    GeometricMorphologyExpert,
    RadiomicsExpertAdapter
)
from app.core.ml_models.wsi_models.spatial_model.SpatialGNNAdapter import SpatialGNNAdapter

# ============================================================
# 1. НАСТРОЙКА ЛОГГИРОВАНИЯ
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# 2. PYDANTIC СХЕМЫ ДЛЯ ЗАПРОСОВ/ОТВЕТОВ
# ============================================================

class UserCreateRequest(BaseModel):
    email: str
    name: str
    password_hash: str
    role: str = "clinician"

class UserLoginRequest(BaseModel):
    email: str
    password_hash: str

class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None

class SlideRegisterRequest(BaseModel):
    project_id: str
    patient_name: str
    tiff_path: str
    parquet_path: str
    hdf5_path: str

class InferenceRequest(BaseModel):
    slide_id: str
    x: int
    y: int
    patch_size: int = 512
    regional_spots: List[dict] = []  # Список спотов с x_local, y_local, spot_id

class TrainingRequest(BaseModel):
    slide_id: str
    epochs: int = 50

class PresetCreateRequest(BaseModel):
    preset_id: str
    name: str
    description: str
    expert_id: str
    spatial_model_id: str

# ============================================================
# 3. ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ
# ============================================================

# --- База данных ---
db_service = DatabaseService(session_factory=lambda: None)  # TODO: подставить реальную сессию SQLAlchemy

# --- Реестр моделей ---
registry = ModelPresetRegistry()

# --- Регистрация экспертов ---
efficientnet_expert = EfficientNetExpertAdapter(
    batch_size=32,
    cache_dir="./data/cache/efficientnet"
)
registry.register_expert("efficientnet", efficientnet_expert)

geometry_expert = GeometricMorphologyExpert(
    k_neighbors=4,
    cache_dir="./data/cache/geometry"
)
registry.register_expert("geometry", geometry_expert)

radiomics_expert = RadiomicsExpertAdapter(
    cache_dir="./data/cache/radiomics"
)
registry.register_expert("radiomics", radiomics_expert)

# --- Регистрация пространственных моделей ---
gene_names = ["CD3D", "CD68", "CD4", "CD8A", "FOXP3", "PDCD1", "CTLA4"]  # Пример генов
spatial_model = SpatialGNNAdapter(
    feature_dim=1280,  # Для EfficientNet
    hidden_dim=256,
    gene_names=gene_names,
    expert_key="EfficientNetExpert",
    spot_radius=50.0
)
registry.register_spatial_model("gnn_v1", spatial_model)

# --- Создание пресетов ---
registry.create_preset(
    preset_id="preset_effnet_gnn",
    name="EfficientNet + GNN",
    description="Морфологический эксперт EfficientNet + GNN",
    expert_id="efficientnet",
    spatial_model_id="gnn_v1"
)

registry.create_preset(
    preset_id="preset_geometry_gnn",
    name="Geometry + GNN",
    description="Геометрический эксперт + GNN",
    expert_id="geometry",
    spatial_model_id="gnn_v1"
)

# Активируем пресет по умолчанию
registry.set_active_preset("preset_effnet_gnn")

# --- Оркестратор ---
orchestrator = XeniumSystemOrchestrator(
    registry=registry,
    db_service=db_service,
    cache_dir="./data/cache/orchestrator"
)

# ============================================================
# 4. FASTAPI ПРИЛОЖЕНИЕ
# ============================================================

app = FastAPI(
    title="Xenium Spatial Transcriptomics Platform API",
    description="API для анализа пространственной транскриптомики с использованием ML",
    version="1.0.0"
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить конкретными доменами
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Простая имитация JWT (в реальном проекте использовать JWT библиотеку)
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Проверяет токен и возвращает пользователя."""
    # TODO: Реализовать настоящую JWT валидацию
    token = credentials.credentials
    # Для демо: любой токен считается валидным
    return {"id": "user_123", "email": "demo@example.com", "role": "admin"}

# ============================================================
# 5. ЭНДПОИНТЫ
# ============================================================

# --- Health Check ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Xenium Spatial Transcriptomics API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# --- Пользователи ---
@app.post("/api/users/register")
def register_user(
    request: UserCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Регистрация нового пользователя (только для admin)."""
    try:
        result = orchestrator.register_user(
            admin_user_id=current_user["id"],
            email=request.email,
            name=request.name,
            password_hash=request.password_hash,
            role=request.role,
            ip_address="127.0.0.1"  # TODO: брать из request.client.host
        )
        return {"success": True, "data": result}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка регистрации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users/login")
def login_user(request: UserLoginRequest):
    """Аутентификация пользователя."""
    user = orchestrator.authenticate_and_get_user(
        email=request.email,
        ip_address="127.0.0.1"
    )
    if not user:
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    
    # Генерируем фейковый JWT токен
    token = f"jwt_{uuid.uuid4().hex}"
    return {
        "success": True,
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role
        }
    }

# --- Проекты ---
@app.post("/api/projects")
def create_project(
    request: ProjectCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Создание нового проекта."""
    try:
        project_id = orchestrator.create_new_project(
            user_id=current_user["id"],
            name=request.name,
            description=request.description,
            ip_address="127.0.0.1"
        )
        return {"success": True, "project_id": project_id}
    except Exception as e:
        logger.error(f"Ошибка создания проекта: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects")
def get_projects(current_user: dict = Depends(get_current_user)):
    """Получение всех проектов пользователя."""
    try:
        data = orchestrator.get_dashboard_data(current_user["id"])
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Ошибка получения проектов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Слайды ---
@app.post("/api/slides/register")
def register_slide(
    request: SlideRegisterRequest,
    current_user: dict = Depends(get_current_user)
):
    """Регистрация нового слайда Xenium."""
    try:
        slide_id = orchestrator.register_and_cache_slide(
            project_id=request.project_id,
            patient_name=request.patient_name,
            tiff_path=request.tiff_path,
            parquet_path=request.parquet_path,
            hdf5_path=request.hdf5_path
        )
        return {"success": True, "slide_id": slide_id}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка регистрации слайда: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/slides/{slide_id}")
def get_slide(slide_id: str, current_user: dict = Depends(get_current_user)):
    """Получение метаданных слайда."""
    try:
        slide_data = orchestrator._ensure_slide_is_cached(slide_id)
        return {
            "success": True,
            "data": {
                "id": slide_id,
                "patient": slide_data["db_record"].patient_name,
                "metadata": slide_data["metadata"].__dict__,
                "status": slide_data["db_record"].status
            }
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка получения слайда: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Инференс ---
@app.post("/api/inference/run")
def run_inference(
    request: InferenceRequest,
    current_user: dict = Depends(get_current_user)
):
    """Запуск инференса на выбранном регионе."""
    try:
        # Конвертируем spots из dict в объекты RegionalSpatialSpot
        regional_spots = []
        for spot_data in request.regional_spots:
            regional_spots.append(RegionalSpatialSpot(
                spot_id=spot_data.get("spot_id", f"spot_{uuid.uuid4().hex[:6]}"),
                x_local=spot_data.get("x_local", 0.0),
                y_local=spot_data.get("y_local", 0.0),
                x_wsi=spot_data.get("x_wsi", 0.0),
                y_wsi=spot_data.get("y_wsi", 0.0)
            ))
        
        result = orchestrator.run_inference_and_save_report(
            user_id=current_user["id"],
            slide_id=request.slide_id,
            x=request.x,
            y=request.y,
            patch_size=request.patch_size,
            regional_spots=regional_spots,
            ip_address="127.0.0.1"
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Ошибка инференса: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Обучение ---
@app.post("/api/training/start")
def start_training(
    request: TrainingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Запуск фонового обучения GNN."""
    try:
        job_id = orchestrator.start_background_training(
            user_id=current_user["id"],
            slide_id=request.slide_id,
            epochs=request.epochs,
            ip_address="127.0.0.1"
        )
        return {"success": True, "job_id": job_id}
    except Exception as e:
        logger.error(f"Ошибка запуска обучения: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/status/{job_id}")
def get_training_status(job_id: str, current_user: dict = Depends(get_current_user)):
    """Получение статуса фонового обучения."""
    status = orchestrator.get_job_status(job_id)
    return {"success": True, "data": status}

# --- Пресеты ---
@app.get("/api/presets")
def get_presets(current_user: dict = Depends(get_current_user)):
    """Получение всех доступных пресетов моделей."""
    try:
        presets = orchestrator.registry.get_available_presets()
        return {
            "success": True,
            "data": [
                {
                    "id": p.preset_id,
                    "name": p.name,
                    "description": p.description,
                    "expert_id": p.expert_id,
                    "spatial_model_id": p.spatial_model_id,
                    "is_active": p.preset_id == orchestrator.registry._active_preset_id
                }
                for p in presets.values()
            ]
        }
    except Exception as e:
        logger.error(f"Ошибка получения пресетов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/presets/activate")
def activate_preset(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Активация пресета."""
    try:
        preset_id = request.get("preset_id")
        if not preset_id:
            raise HTTPException(status_code=400, detail="preset_id обязателен")
        
        orchestrator.registry.set_active_preset(preset_id)
        return {"success": True, "active_preset": preset_id}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка активации пресета: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Отчеты ---
@app.get("/api/reports/slide/{slide_id}")
def get_slide_reports(
    slide_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Получение всех отчетов для слайда."""
    try:
        reports = db_service.get_slide_reports(slide_id)
        return {
            "success": True,
            "data": [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat(),
                    "roi_coordinates": r.roi_coordinates,
                    "predicted_data": r.predicted_expression_data
                }
                for r in reports
            ]
        }
    except Exception as e:
        logger.error(f"Ошибка получения отчетов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/{report_id}")
def get_report(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Получение отчета по ID."""
    try:
        report = db_service.get_report_by_id(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Отчет не найден")
        
        return {
            "success": True,
            "data": {
                "id": report.id,
                "created_at": report.created_at.isoformat(),
                "roi_coordinates": report.roi_coordinates,
                "predicted_expression": report.predicted_expression_data
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения отчета: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 6. ЗАПУСК СЕРВЕРА
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
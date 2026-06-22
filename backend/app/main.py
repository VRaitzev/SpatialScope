import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Generator, List, Optional
from fastapi import Request, Query, HTTPException, Depends
from sqlalchemy.orm import Session
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.data_managers.ORM_manager import DatabaseService
from backend.app.core.data_managers.data_schemes import Base
from app.core.ml_models.wsi_models.expert_models.EfficientNetExpertAdapter import (
    EfficientNetExpertAdapter,
)
from app.core.ml_models.wsi_models.expert_models.GeometricMorphologyExpert import (
    GeometricMorphologyExpert,
)
from app.core.ml_models.wsi_models.expert_models.RadiomicsExpertAdapter import (
    RadiomicsExpertAdapter,
)
from app.core.ml_models.wsi_models.spatial_model.SpatialGNNAdapter import (
    SpatialGNNAdapter,
)
from app.core.orchestrators.XeniumSystemOrchestrator import XeniumSystemOrchestrator
from app.core.registry.ModelPresetRegistry import ModelPresetRegistry
from app.core.schemes.Cells import RegionalSpatialSpot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Settings:
    SECRET_KEY = os.getenv(
        "SECRET_KEY", "your-secret-key-change-in-production-1234567890"
    )
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./xenium_system.db")


settings = Settings()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Недействительный токен") from exc


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.DATABASE_URL
    else {},
)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


from fastapi import Query

def get_current_user(
    request: Request,
    token: Optional[str] = Query(None), # Токен из URL (для OpenSeadragon)
    db: Session = Depends(get_db),
) -> dict:
    # 1. Пытаемся достать токен из заголовка Authorization
    auth_header = request.headers.get("Authorization")
    token_str = None

    if auth_header and auth_header.startswith("Bearer "):
        token_str = auth_header.split(" ")[1]
    # 2. Если в заголовке нет, берем из URL
    elif token:
        token_str = token

    if not token_str:
        raise HTTPException(status_code=401, detail="Токен отсутствует")

    # 3. Декодируем
    payload = decode_token(token_str)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Неверный токен")

    # 4. Ищем в БД
    db_service = DatabaseService(lambda: db)
    user = db_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")

    return {"id": user.id, "email": user.email, "name": user.name, "role": user.role}


def require_roles(*allowed_roles: str):
    def dependency(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=403, detail="Недостаточно прав для выполнения операции"
            )
        return current_user

    return dependency


class UserCreateRequest(BaseModel):
    email: str
    name: str
    password: str
    role: str = "clinician"


class UserLoginRequest(BaseModel):
    email: str
    password: str


class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None


class SlideRegisterRequest(BaseModel):
    project_id: str
    patient_name: str
    tiff_path: str
    parquet_path: str
    hdf5_path: str


class RegionRequest(BaseModel):
    x: int
    y: int
    width: int
    height: int


class InferenceRequest(BaseModel):
    slide_id: str
    x: int = 0
    y: int = 0
    patch_size: int = 512
    preset_id: Optional[str] = None
    inference_region: Optional[RegionRequest] = None
    regional_spots: List[dict] = Field(default_factory=list)
    target_gene: Optional[str] = None
    include_interpretation: bool = True


class TrainingRequest(BaseModel):
    slide_id: str
    preset_id: Optional[str] = None
    epochs: int = 50
    patch_size: int = 1024
    max_patches: int = 20
    training_region: Optional[RegionRequest] = None


class PresetActivateRequest(BaseModel):
    preset_id: str


class InterpretationRequest(BaseModel):
    target_gene: Optional[str] = None


# ---------------------------------------------------------------------------
# System bootstrap
# ---------------------------------------------------------------------------

db_service = DatabaseService(lambda: SessionLocal())
registry = ModelPresetRegistry()
gene_names = ["CD3D", "CD68", "CD4", "CD8A", "FOXP3", "PDCD1", "CTLA4"]

for expert_id, factory in [
    (
        "efficientnet",
        lambda: EfficientNetExpertAdapter(
            batch_size=32, cache_dir="./data/cache/efficientnet"
        ),
    ),
    (
        "geometry",
        lambda: GeometricMorphologyExpert(
            k_neighbors=4, cache_dir="./data/cache/geometry"
        ),
    ),
    ("radiomics", lambda: RadiomicsExpertAdapter(cache_dir="./data/cache/radiomics")),
]:
    try:
        registry.register_expert(expert_id, factory())
        logger.info("Expert registered: %s", expert_id)
    except Exception as exc:
        logger.warning("Expert %s not loaded: %s", expert_id, exc)

for model_id, feature_dim in [("gnn_multi", 1386), ("gnn_radiomics", 107)]:
    try:
        registry.register_spatial_model(
            model_id,
            SpatialGNNAdapter(
                feature_dim=feature_dim,
                hidden_dim=256,
                gene_names=gene_names,
                expert_key="placeholder",
                spot_radius=50.0,
            ),
        )
        logger.info("Spatial model registered: %s", model_id)
    except Exception as exc:
        logger.warning("Spatial model %s not loaded: %s", model_id, exc)

try:
    registry.create_preset(
        preset_id="preset_multi_expert_gnn",
        name="EfficientNet + Geometry + Radiomics + GNN",
        description="Multi-expert morphological profile with spatial GNN",
        expert_ids=["efficientnet", "geometry", "radiomics"],
        spatial_model_id="gnn_multi",
    )
    registry.create_preset(
        preset_id="preset_radiomics_gnn",
        name="Radiomics + GNN",
        description="Radiomics-only expert with dedicated GNN",
        expert_ids=["radiomics"],
        spatial_model_id="gnn_radiomics",
    )
    registry.set_active_preset("preset_multi_expert_gnn")
except Exception as exc:
    logger.warning("Preset initialization failed: %s", exc)

orchestrator = XeniumSystemOrchestrator(
    registry=registry,
    db_service=db_service,
    cache_dir="./data/cache/orchestrator",
)

app = FastAPI(
    title="Xenium Spatial Transcriptomics Platform API",
    description="API для анализа пространственной транскриптомики с ML и интерпретацией",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Укажи точный адрес фронтенда, никаких "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Xenium Spatial Transcriptomics API",
        "version": "2.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_preset": registry._active_preset_id,
        "presets": list(registry.get_available_presets().keys()),
    }


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


@app.post("/api/users/register")
def register_user(request: UserCreateRequest, db: Session = Depends(get_db)):
    try:
        db_service_local = DatabaseService(lambda: db)
        if db_service_local.get_user_by_email(request.email):
            raise HTTPException(status_code=400, detail="Email уже зарегистрирован")

        user = db_service_local.create_user(
            email=request.email,
            name=request.name,
            password_hash=get_password_hash(request.password),
            role=request.role,
        )
        return {
            "success": True,
            "data": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Registration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/users/login")
def login_user(request: UserLoginRequest, db: Session = Depends(get_db)):
    try:
        db_service_local = DatabaseService(lambda: db)
        user = db_service_local.get_user_by_email(request.email)
        if not user:
            raise HTTPException(status_code=401, detail="Неверный email или пароль")

        stored_hash = str(user.password_hash)
        if not verify_password(request.password, stored_hash):
            raise HTTPException(status_code=401, detail="Неверный email или пароль")

        token = create_access_token(
            {"sub": user.id, "email": user.email, "role": user.role}
        )
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Login error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/users/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {"success": True, "data": current_user}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


@app.post("/api/projects")
def create_project(
    request: ProjectCreateRequest,
    current_user: dict = Depends(require_roles("admin", "researcher")),
):
    try:
        project_id = orchestrator.create_new_project(
            user_id=current_user["id"],
            name=request.name,
            description=request.description or None,
            ip_address="127.0.0.1",
        )
        return {"success": True, "project_id": project_id}
    except Exception as exc:
        logger.error("Project creation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/projects")
def get_projects(current_user: dict = Depends(get_current_user)):
    try:
        data = orchestrator.get_dashboard_data(current_user["id"])
        return {"success": True, "data": data}
    except Exception as exc:
        logger.error("Projects fetch error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------


@app.post("/api/slides/register")
def register_slide(
    request: SlideRegisterRequest,
    current_user: dict = Depends(require_roles("admin", "researcher")),
):
    try:
        slide_id = orchestrator.register_and_cache_slide(
            project_id=request.project_id,
            patient_name=request.patient_name,
            tiff_path=request.tiff_path,
            parquet_path=request.parquet_path,
            hdf5_path=request.hdf5_path,
        )
        return {"success": True, "slide_id": slide_id}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Slide registration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/slides/{slide_id}")
def get_slide(slide_id: str, current_user: dict = Depends(get_current_user)):
    try:
        data = orchestrator.get_slide_metadata(slide_id)
        return {"success": True, "data": data}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Slide fetch error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/slides/{slide_id}/image")
def get_slide_image(slide_id: str, current_user: dict = Depends(get_current_user)):
    try:
        meta = orchestrator.get_slide_metadata(slide_id)
        tiff_path = meta["tiff_path"]
        if not os.path.exists(tiff_path):
            raise HTTPException(status_code=404, detail="Файл слайда не найден")

        slide_data = orchestrator._ensure_slide_is_cached(slide_id)
        wsi_loader = slide_data["wsi_loader"]
        content = wsi_loader.build_thumbnail_bytes(tiff_path)
        return Response(content=content, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/slides/{slide_id}/patch")
def get_slide_patch(
    slide_id: str,
    x: int,
    y: int,
    width: int,
    height: int,
    current_user: dict = Depends(get_current_user),
):
    try:
        meta = orchestrator.get_slide_metadata(slide_id)
        tiff_path = meta["tiff_path"]
        if not os.path.exists(tiff_path):
            raise HTTPException(status_code=404, detail="Файл слайда не найден")

        slide_data = orchestrator._ensure_slide_is_cached(slide_id)
        wsi_loader = slide_data["wsi_loader"]
        content = wsi_loader.build_patch_bytes(
            tiff_path,
            x=x,
            y=y,
            width=width,
            height=height,
        )
        return Response(content=content, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Inference & training
# ---------------------------------------------------------------------------


@app.post("/api/inference/run")
def run_inference(
    request: InferenceRequest,
    current_user: dict = Depends(require_roles("admin", "researcher", "clinician")),
):
    try:
        regional_spots = [
            RegionalSpatialSpot(
                spot_id=spot.get("spot_id", f"spot_{uuid.uuid4().hex[:6]}"),
                x_local=spot.get("x_local", 0.0),
                y_local=spot.get("y_local", 0.0),
                x_wsi=spot.get("x_wsi", 0.0),
                y_wsi=spot.get("y_wsi", 0.0),
                target_expressions=spot.get("target_expressions"),
            )
            for spot in request.regional_spots
        ]

        result = orchestrator.run_inference_and_save_report(
            user_id=current_user["id"],
            slide_id=request.slide_id,
            x=request.x,
            y=request.y,
            patch_size=request.patch_size,
            regional_spots=regional_spots,
            ip_address="127.0.0.1",
            preset_id=request.preset_id,
            inference_region=request.inference_region.model_dump()
            if request.inference_region
            else None,
            target_gene=request.target_gene,
            include_interpretation=request.include_interpretation,
        )
        return {"success": True, "data": result}
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/training/start")
def start_training(
    request: TrainingRequest,
    current_user: dict = Depends(require_roles("admin", "researcher")),
):
    try:
        job_id = orchestrator.start_background_training(
            user_id=current_user["id"],
            slide_id=request.slide_id,
            epochs=request.epochs,
            ip_address="127.0.0.1",
            preset_id=request.preset_id,
            training_region=request.training_region.model_dump()
            if request.training_region
            else None,
            patch_size=request.patch_size,
            max_patches=request.max_patches,
        )
        return {"success": True, "job_id": job_id}
    except Exception as exc:
        logger.error("Training start error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/training/status/{job_id}")
def get_training_status(job_id: str, current_user: dict = Depends(get_current_user)):
    status_data = orchestrator.get_job_status(job_id)
    return {"success": True, "data": status_data}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@app.get("/api/presets")
def get_presets(current_user: dict = Depends(get_current_user)):
    try:
        presets = registry.get_available_presets()
        return {
            "success": True,
            "data": [
                {
                    "id": p.preset_id,
                    "name": p.name,
                    "description": p.description,
                    "expert_ids": p.expert_ids,
                    "expert_id": ", ".join(p.expert_ids),
                    "spatial_model_id": p.spatial_model_id,
                    "composite_expert_key": p.composite_expert_key,
                    "is_active": p.preset_id == registry._active_preset_id,
                    "weights_path": registry.get_weights_path(p.preset_id),
                    "weights_exists": os.path.exists(
                        registry.get_weights_path(p.preset_id)
                    ),
                }
                for p in presets.values()
            ],
        }
    except Exception as exc:
        logger.error("Presets fetch error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/presets/activate")
def activate_preset(
    request: PresetActivateRequest,
    current_user: dict = Depends(require_roles("admin")),
):
    try:
        registry.set_active_preset(request.preset_id)
        return {"success": True, "active_preset": request.preset_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Preset activation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Reports & interpretation
# ---------------------------------------------------------------------------


@app.get("/api/reports/slide/{slide_id}")
def get_slide_reports(
    slide_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        db_service_local = DatabaseService(lambda: db)
        reports = db_service_local.get_slide_reports(slide_id)
        return {
            "success": True,
            "data": [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat(),
                    "preset_id": r.preset_id,
                    "roi_coordinates": r.roi_coordinates,
                    "predicted_data": r.predicted_expression_data,
                }
                for r in reports
            ],
        }
    except Exception as exc:
        logger.error("Reports fetch error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/reports/{report_id}")
def get_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        db_service_local = DatabaseService(lambda: db)
        report = db_service_local.get_report_by_id(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Отчет не найден")

        return {
            "success": True,
            "data": {
                "id": report.id,
                "created_at": report.created_at.isoformat(),
                "preset_id": report.preset_id,
                "roi_coordinates": report.roi_coordinates,
                "predicted_expression": report.predicted_expression_data,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Report fetch error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/reports/{report_id}/interpret")
def interpret_report(
    report_id: str,
    request: InterpretationRequest = InterpretationRequest(),
    current_user: dict = Depends(get_current_user),
):
    try:
        result = orchestrator.interpret_report(
            report_id, target_gene=request.target_gene
        )
        return {"success": True, "data": result}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Interpretation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/reports/{report_id}/export/json")
def export_report_json(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    db_service_local = DatabaseService(lambda: db)
    report = db_service_local.get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")
    return JSONResponse(
        content={
            "id": report.id,
            "created_at": report.created_at.isoformat(),
            "preset_id": report.preset_id,
            "roi_coordinates": report.roi_coordinates,
            "predicted_expression": report.predicted_expression_data,
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )

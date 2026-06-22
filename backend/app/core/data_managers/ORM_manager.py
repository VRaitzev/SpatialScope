import logging
from typing import Any, Dict, List, Optional, cast

from sqlalchemy.orm import Session

from backend.app.core.data_managers.data_schemes import (
    AnalysisReport,
    AuditLog,
    MLModel,
    ModelPreset,
    Project,
    User,
    XeniumSlide,
)


class DatabaseService:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.logger = logging.getLogger("DatabaseService")
        logging.basicConfig(level=logging.INFO)

    def _execute(self, func, *args, **kwargs):
        """Вспомогательный метод-управленец жизненным циклом сессии (Pattern: Execute Around)."""
        session = self.session_factory()
        try:
            result = func(session, *args, **kwargs)
            session.expunge_all()
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            self.logger.error(f"❌ [Database Error] Крах транзакции: {e}")
            raise e
        finally:
            session.close()

    def create_user(
        self, email: str, name: str, password_hash: str, role: str = "clinician"
    ) -> User:
        def _op(s: Session):
            user = User(email=email, name=name, password_hash=password_hash, role=role)
            s.add(user)
            s.flush()
            return user

        return self._execute(_op)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Критически важно для авторизации (JWT Login)"""
        return self._execute(
            lambda s: s.query(User).filter(User.email == email).first()
        )

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        return self._execute(lambda s: s.query(User).filter(User.id == user_id).first())

    def create_project(
        self, user_id: str, name: str, description: Optional[str] = None
    ) -> Project:
        def _op(s: Session):
            proj = Project(user_id=user_id, name=name, description=description)
            s.add(proj)
            s.flush()
            return proj

        return self._execute(_op)

    def get_project_by_id(self, project_id: str) -> Optional[Project]:
        return self._execute(
            lambda s: s.query(Project).filter(Project.id == project_id).first()
        )

    def get_user_projects(self, user_id: str) -> List[Project]:
        return self._execute(
            lambda s: s.query(Project).filter(Project.user_id == user_id).all()
        )

    def delete_project(self, project_id: str) -> bool:
        return self._execute(
            lambda s: s.query(Project).filter(Project.id == project_id).delete() > 0
        )

    def register_xenium_slide(
        self,
        project_id: str,
        patient_name: str,
        tiff_path: str,
        parquet_path: str,
        hdf5_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> XeniumSlide:
        """Регистрирует тройку путей к файлам Xenium, необходимую оркестратору."""

        def _op(s: Session):
            slide = XeniumSlide(
                project_id=project_id,
                patient_name=patient_name,
                slide_tiff_path=tiff_path,
                cells_parquet_path=parquet_path,
                matrix_hdf5_path=hdf5_path,
                slide_metadata=metadata,
                status="ready",
            )
            s.add(slide)
            s.flush()
            return slide

        return self._execute(_op)

    def get_slide_by_id(self, slide_id: str) -> Optional[XeniumSlide]:
        return self._execute(
            lambda s: s.query(XeniumSlide).filter(XeniumSlide.id == slide_id).first()
        )

    def get_project_slides(self, project_id: str) -> List[XeniumSlide]:
        return self._execute(
            lambda s: (
                s.query(XeniumSlide).filter(XeniumSlide.project_id == project_id).all()
            )
        )

    def update_slide_status(self, slide_id: str, status: str) -> bool:
        def _op(s: Session):
            slide = s.query(XeniumSlide).filter(XeniumSlide.id == slide_id).first()
            if slide:
                slide.status = cast(Any, status)
            return slide is not None

        return self._execute(_op)

    def register_ml_model(
        self,
        name: str,
        model_type: str,
        model_key: str,
        weights_path: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> MLModel:
        """Добавляет в систему базовую атомарную модель (например, новые веса GNN)"""

        def _op(s: Session):
            model = MLModel(
                name=name,
                model_type=model_type,
                model_key=model_key,
                weights_path=weights_path,
                manifest=manifest,
            )
            s.add(model)
            s.flush()
            return model

        return self._execute(_op)

    def create_model_preset(
        self, name: str, description: str, expert_model_id: str, spatial_model_id: str
    ) -> ModelPreset:
        """Сшивает две нейросети в один готовый диагностический пресет"""

        def _op(s: Session):
            preset = ModelPreset(
                name=name,
                description=description,
                expert_model_id=expert_model_id,
                spatial_model_id=spatial_model_id,
            )
            s.add(preset)
            s.flush()
            return preset

        return self._execute(_op)

    def get_all_presets(self) -> List[ModelPreset]:
        return self._execute(lambda s: s.query(ModelPreset).all())

    def get_preset_by_id(self, preset_id: str) -> Optional[ModelPreset]:
        return self._execute(
            lambda s: s.query(ModelPreset).filter(ModelPreset.id == preset_id).first()
        )

    def create_analysis_report(
        self,
        user_id: str,
        slide_id: str,
        preset_id: str,
        roi_coordinates: Dict[str, Any],
        predicted_data: Dict[str, Any],
    ) -> AnalysisReport:
        """
        Сохраняет результаты работы run_inference_on_active_preset в историю пациента.
        Сюда передаются сериализованные Pydantic-схемы!
        """

        def _op(s: Session):
            report = AnalysisReport(
                user_id=user_id,
                slide_id=slide_id,
                preset_id=preset_id,
                roi_coordinates=roi_coordinates,
                predicted_expression_data=predicted_data,
            )
            s.add(report)
            s.flush()
            return report

        return self._execute(_op)

    def get_report_by_id(self, report_id: str) -> Optional[AnalysisReport]:
        return self._execute(
            lambda s: (
                s.query(AnalysisReport).filter(AnalysisReport.id == report_id).first()
            )
        )

    def get_slide_reports(self, slide_id: str) -> List[AnalysisReport]:
        """Позволяет Врачу посмотреть историю всех анализов по конкретному пациенту"""
        return self._execute(
            lambda s: (
                s.query(AnalysisReport)
                .filter(AnalysisReport.slide_id == slide_id)
                .order_by(AnalysisReport.created_at.desc())
                .all()
            )
        )

    def log_action(
        self, user_id: Optional[str], action: str, ip_address: Optional[str] = None
    ) -> None:
        """Запись критических событий безопасности в системе"""
        self._execute(
            lambda s: s.add(
                AuditLog(user_id=user_id, action=action, ip_address=ip_address)
            )
        )

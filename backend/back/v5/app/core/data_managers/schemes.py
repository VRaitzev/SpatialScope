import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker

Base = declarative_base()

# =====================================================================
# 1. СЛОЙ ПОЛЬЗОВАТЕЛЕЙ И ДОСТУПА
# =====================================================================

class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    
    # Роли разделяем строго под наши сценарии: 'clinician' (Врач), 'researcher' (ML-инженер), 'admin'
    role = Column(String, default='clinician', nullable=False)  
    
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    reports = relationship("AnalysisReport", back_populates="clinician")


class Project(Base):
    """Контейнер исследований (например, 'Кафедра Онкологии - Скрининг 2026')"""
    __tablename__ = 'projects'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id', ondelete="CASCADE"))
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="projects")
    slides = relationship("XeniumSlide", back_populates="project", cascade="all, delete-orphan")


# =====================================================================
# 2. СЛОЙ МУЛЬТИМОДАЛЬНЫХ ДАННЫХ XENIUM
# =====================================================================

class XeniumSlide(Base):
    """
    Таблица для хранения путей к Xenium-слайдам.
    Решает проблему хранения трех разнородных файлов для одного пациента.
    """
    __tablename__ = 'xenium_slides'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey('projects.id', ondelete="CASCADE"))
    patient_name = Column(String, nullable=False)
    
    # Комплекс путей (в S3 или локальном хранилище), который требует наш Оркестратор
    slide_tiff_path = Column(String, nullable=False)      # Исходное WSI изображение (.ome.tif)
    cells_parquet_path = Column(String, nullable=False)   # Координаты сегментации клеток
    matrix_hdf5_path = Column(String, nullable=False)     # Матрица транскриптомной экспрессии
    
    # Кэшируем метаданные слайда (размеры, MPP, каналы), чтобы не читать тяжелый файл каждый раз
    slide_metadata = Column(JSON, nullable=True) 
    
    status = Column(String, default='queued')  # queued, processing, ready, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="slides")
    reports = relationship("AnalysisReport", back_populates="slide", cascade="all, delete-orphan")


# =====================================================================
# 3. СЛОЙ УПРАВЛЕНИЯ ПРЕСЕТАМИ МОДЕЛЕЙ (Для Реестра)
# =====================================================================

class MLModel(Base):
    """Реестр атомарных ML-моделей (загруженных весов)"""
    __tablename__ = 'ml_models'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    
    # 'expert' (EfficientNet/Radiomics) или 'spatial' (GNN/Transformer)
    model_type = Column(String, nullable=False)  
    
    # Ключ, по которому модель опознается в коде оркестратора (например, 'EfficientNetExpert')
    model_key = Column(String, nullable=False)   
    weights_path = Column(String, nullable=True)  # Путь к файлу .pth или .pkl на сервере
    manifest = Column(JSON, nullable=True)        # Паспорт модели (версия, описание генов)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelPreset(Base):
    """
    Таблица Пресетов. Связывает эксперта 1-го уровня и GNN 2-го уровня в один кликабельный пак.
    Именно этот пресет Врач выбирает на фронтенде!
    """
    __tablename__ = 'model_presets'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(String)
    
    # Связи с таблицей атомарных моделей
    expert_model_id = Column(String, ForeignKey('ml_models.id'))
    spatial_model_id = Column(String, ForeignKey('ml_models.id'))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Настраиваем явные связи с именами (используем инородные имена для однозначности)
    expert_model = relationship("MLModel", foreign_keys=[expert_model_id])
    spatial_model = relationship("MLModel", foreign_keys=[spatial_model_id])


# =====================================================================
# 4. СЛОЙ КЛИНИЧЕСКИХ ОТЧЕТОВ И АУДИТА
# =====================================================================

class AnalysisReport(Base):
    """
    Таблица клинических заключений.
    Сюда складывается весь результат работы метода run_inference_on_active_preset.
    """
    __tablename__ = 'analysis_reports'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'))
    slide_id = Column(String, ForeignKey('xenium_slides.id', ondelete="CASCADE"))
    preset_id = Column(String, ForeignKey('model_presets.id'))
    
    # Координаты патча (какой именно участок ткани анализировал врач)
    roi_coordinates = Column(JSON, nullable=False)  # {"x": 1024, "y": 2048, "patch_size": 512}
    
    # Самое главное: Предсказанный профиль экспрессии (структура RegionalPredictionOutput в JSON)
    predicted_expression_data = Column(JSON, nullable=False) 
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    clinician = relationship("User", back_populates="reports")
    slide = relationship("XeniumSlide", back_populates="reports")
    preset = relationship("ModelPreset")


class AuditLog(Base):
    """Логирование действий для ИСТ-безопасности системы"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id', ondelete="SET NULL"), nullable=True)
    action = Column(String, nullable=False)  # например: 'USER_LOGIN', 'RUN_INFERENCE', 'CREATE_PRESET'
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)


# --- Инициализация локальной БД ---
if __name__ == "__main__":
    engine = create_engine('sqlite:///system_data.db')
    Base.metadata.create_all(engine)
    print("🚀 [Database] База данных успешно расширена, все новые таблицы и связи созданы!")
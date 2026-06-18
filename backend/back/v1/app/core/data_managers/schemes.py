from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default='researcher')  # 'admin' или 'researcher'
    projects = relationship("Project", back_populates="owner")

class Project(Base):
    __tablename__ = 'projects'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'))
    name = Column(String, nullable=False)
    wsi_s3_path = Column(String, nullable=False)
    target_genes = Column(JSON)  # Список генов в формате JSON
    status = Column(String, default='queued')  # queued, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="projects")

class ModelConfig(Base):
    __tablename__ = 'model_configs'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    params = Column(JSON)  # Гиперпараметры модели

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'))
    action = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)

# Инициализация БД
engine = create_engine('sqlite:///system_data.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
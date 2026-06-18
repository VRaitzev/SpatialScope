import logging
from datetime import datetime
from sqlalchemy.orm import Session
from schemes import User, Project, AuditLog


class DatabaseService:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.logger = logging.getLogger("DatabaseService")
        logging.basicConfig(level=logging.INFO)

    def _execute(self, func, *args, **kwargs):
        """Вспомогательный метод-обертка для управления сессиями (шаблон 'Execute Around')."""
        session = self.session_factory()
        try:
            result = func(session, *args, **kwargs)
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            self.logger.error(f"Ошибка транзакции БД: {e}")
            raise e
        finally:
            session.close()

    # --- CREATE ---
    def create_user(self, email, name, password_hash, role='researcher'):
        return self._execute(lambda s: s.add(User(email=email, name=name, password_hash=password_hash, role=role)) or True)

    def create_project(self, user_id, name, wsi_path, target_genes):
        def _op(s):
            proj = Project(user_id=user_id, name=name, wsi_s3_path=wsi_path, target_genes=target_genes)
            s.add(proj)
            s.flush()
            s.refresh(proj)
            return proj.id
        return self._execute(_op)

    # --- READ ---
    def get_project_by_id(self, project_id):
        return self._execute(lambda s: s.query(Project).filter(Project.id == project_id).first())

    def get_user_projects(self, user_id):
        return self._execute(lambda s: s.query(Project).filter(Project.user_id == user_id).all())

    # --- UPDATE ---
    def update_project_status(self, project_id, status):
        def _op(s):
            proj = s.query(Project).filter(Project.id == project_id).first()
            if proj:
                proj.status = status
            return proj is not None
        return self._execute(_op)

    def update_project_fields(self, project_id, **kwargs):
        return self._execute(lambda s: s.query(Project).filter(Project.id == project_id).update(kwargs))

    # --- DELETE ---
    def delete_project(self, project_id):
        return self._execute(lambda s: s.query(Project).filter(Project.id == project_id).delete() > 0)

    # --- LOGGING (для диплома) ---
    def log_action(self, user_id, action, ip_address=None):
        return self._execute(lambda s: s.add(AuditLog(user_id=user_id, action=action, ip_address=ip_address)))
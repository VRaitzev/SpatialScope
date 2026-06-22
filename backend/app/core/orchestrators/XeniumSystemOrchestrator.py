import logging
import os
import threading
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import torch

from app.core.data_managers.ORM_manager import DatabaseService
from app.core.interpretability.InterpretationService import InterpretationService
from app.core.loaders.connector.XeniumGraphDataset import XeniumGraphDataset
from app.core.loaders.rna.XeniumGenePanelLoader import XeniumGenePanelLoader
from app.core.loaders.wsi.image.XeniumWSILoader import XeniumWSILoader
from app.core.loaders.wsi.segmentatioin.XeniumCellBoundaryAdapter import (
    XeniumCellBoundaryAdapter,
)
from app.core.ml_models.wsi_models.expert_models.CompositeExpertAdapter import (
    CompositeExpertAdapter,
)
from app.core.ml_models.wsi_models.expert_models.EfficientNetExpertAdapter import (
    EfficientNetExpertAdapter,
)
from app.core.ml_models.wsi_models.expert_models.RadiomicsExpertAdapter import (
    RadiomicsExpertAdapter,
)
from app.core.pipeline.TrainPipeline import SpatialTrainPipeline
from app.core.registry.ModelPresetRegistry import ModelPresetRegistry
from app.core.schemes.Cells import (
    RegionalPredictionOutput,
    RegionalSpatialSpot,
    WSICellFeatureRegistry,
)
from app.core.schemes.Segmentation import SegmentationContainer
from app.core.schemes.WSI import WSIContainer
from app.core.utils.region_utils import SlideRegion, iter_patch_coords

if TYPE_CHECKING:
    from backend.app.core.data_managers.data_schemes import AnalysisReport, Project, XeniumSlide

logger = logging.getLogger(__name__)


class XeniumSystemOrchestrator:
    """Main coordinator linking API, ML engine, loaders and database."""

    def __init__(
        self,
        registry: ModelPresetRegistry,
        db_service: DatabaseService,
        cache_dir: str = "./data/cache",
    ):
        self.cache_dir = cache_dir
        self.registry = registry
        self.db = db_service
        self.interpretation_service = InterpretationService(registry)
        os.makedirs(cache_dir, exist_ok=True)

        self.slides_cache: Dict[str, Dict[str, Any]] = {}
        self.jobs_registry: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Slide management
    # ------------------------------------------------------------------

    def register_and_cache_slide(
        self,
        project_id: str,
        patient_name: str,
        tiff_path: str,
        parquet_path: str,
        hdf5_path: str,
    ) -> str:
        for path, name in [
            (tiff_path, "TIFF"),
            (parquet_path, "Parquet"),
            (hdf5_path, "HDF5"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл {name} не найден: {path}")

        gene_loader = XeniumGenePanelLoader(matrix_path=hdf5_path)
        wsi_loader = XeniumWSILoader(cells_path=parquet_path)
        boundary_adapter = cast(
            Any,
            XeniumCellBoundaryAdapter(
                xenium_loader=wsi_loader, gene_loader=gene_loader
            ),
        )

        metadata: WSIContainer = wsi_loader.load_slide_metadata(tiff_path)
        wsi_loader.apply_registration_to_dataframe(reg_service=None)

        metadata_dict = {
            "width": metadata.dimensions_level_0[0],
            "height": metadata.dimensions_level_0[1],
            "staining": metadata.staining,
            "mpp": metadata.mpp,
        }

        db_slide = cast(
            "XeniumSlide",
            self.db.register_xenium_slide(
                project_id=project_id,
                patient_name=patient_name,
                tiff_path=tiff_path,
                parquet_path=parquet_path,
                hdf5_path=hdf5_path,
                metadata=metadata_dict,
            ),
        )

        slide_id = str(db_slide.id)
        self.slides_cache[slide_id] = {
            "wsi_loader": wsi_loader,
            "gene_loader": gene_loader,
            "boundary_adapter": boundary_adapter,
            "metadata": metadata,
            "db_record": db_slide,
        }
        return slide_id

    def _ensure_slide_is_cached(self, slide_id: str) -> Dict[str, Any]:
        if slide_id in self.slides_cache:
            return self.slides_cache[slide_id]

        db_slide = cast("XeniumSlide | None", self.db.get_slide_by_id(slide_id))
        if not db_slide:
            raise KeyError(f"Слайд {slide_id} не найден")

        gene_loader = XeniumGenePanelLoader(matrix_path=str(db_slide.matrix_hdf5_path))
        wsi_loader = XeniumWSILoader(cells_path=str(db_slide.cells_parquet_path))
        boundary_adapter = cast(
            Any,
            XeniumCellBoundaryAdapter(
                xenium_loader=wsi_loader, gene_loader=gene_loader
            ),
        )

        metadata = wsi_loader.load_slide_metadata(str(db_slide.slide_tiff_path))
        wsi_loader.apply_registration_to_dataframe(reg_service=None)

        self.slides_cache[slide_id] = {
            "wsi_loader": wsi_loader,
            "gene_loader": gene_loader,
            "boundary_adapter": boundary_adapter,
            "metadata": metadata,
            "db_record": db_slide,
        }
        return self.slides_cache[slide_id]

    def get_slide_metadata(self, slide_id: str) -> Dict[str, Any]:
        slide_data = self._ensure_slide_is_cached(slide_id)
        meta = slide_data["metadata"]
        db = slide_data["db_record"]
        return {
            "id": slide_id,
            "patient": db.patient_name,
            "status": db.status,
            "width": meta.dimensions_level_0[0],
            "height": meta.dimensions_level_0[1],
            "staining": meta.staining,
            "mpp": meta.mpp,
            "tiff_path": db.slide_tiff_path,
        }

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _patch_tensor(self, patch_region) -> torch.Tensor:
        if hasattr(patch_region, "image_tensor"):
            return patch_region.image_tensor
        if hasattr(patch_region, "image"):
            img = patch_region.image
            if hasattr(img, "shape"):
                return torch.from_numpy(img).permute(2, 0, 1)
            return img
        raise AttributeError("Patch region has no image data")

    def _run_expert_on_patch(
        self,
        preset_id: str,
        patch_region,
        segmentation: SegmentationContainer,
        patch_id: str,
        slide_id: str,
    ) -> WSICellFeatureRegistry:
        expert_model, _ = self.registry.get_models_for_preset(preset_id)

        if isinstance(expert_model, CompositeExpertAdapter):
            return expert_model.predict(
                patch_image=self._patch_tensor(patch_region),
                cell_registry=segmentation,
                patch_id=patch_id,
                slide_id=slide_id,
            )

        if isinstance(
            expert_model, (EfficientNetExpertAdapter, RadiomicsExpertAdapter)
        ):
            return expert_model.predict(
                patch_image=self._patch_tensor(patch_region),
                cell_registry=segmentation,
                patch_id=patch_id,
            )

        return expert_model.predict(
            cell_registry=segmentation,
        )

    def _process_patch(
        self,
        slide_data: Dict[str, Any],
        slide_id: str,
        preset_id: str,
        x: int,
        y: int,
        patch_size: int,
    ) -> Tuple[Optional[WSICellFeatureRegistry], Optional[Any], str]:
        wsi_loader: XeniumWSILoader = slide_data["wsi_loader"]
        adapter: XeniumCellBoundaryAdapter = slide_data["boundary_adapter"]
        metadata: WSIContainer = slide_data["metadata"]

        patch_id = f"patch_{slide_id}_{x}_{y}_{patch_size}"
        patch_region = wsi_loader.load_region(metadata, x, y, patch_size, patch_size)
        segmentation: SegmentationContainer = adapter.load_boundaries(
            patch_region, registration_service=None
        )

        if not segmentation.instances:
            return None, patch_region, patch_id

        registry = self._run_expert_on_patch(
            preset_id, patch_region, segmentation, patch_id, slide_id
        )
        registry.x_wsi = float(x)
        registry.y_wsi = float(y)
        registry.slide_id = slide_id
        return registry, patch_region, patch_id

    def _resolve_preset_id(self, preset_id: Optional[str]) -> str:
        if preset_id:
            self.registry.get_preset(preset_id)
            return preset_id
        _, _ = self.registry.get_active_models()
        return cast(str, self.registry._active_preset_id)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference_and_save_report(
        self,
        user_id: str,
        slide_id: str,
        x: int,
        y: int,
        patch_size: int,
        regional_spots: List[RegionalSpatialSpot],
        ip_address: Optional[str] = None,
        preset_id: Optional[str] = None,
        inference_region: Optional[Dict] = None,
        target_gene: Optional[str] = None,
        include_interpretation: bool = True,
    ) -> Dict[str, Any]:
        preset_id = self._resolve_preset_id(preset_id)
        slide_data = self._ensure_slide_is_cached(slide_id)
        _, spatial_model = self.registry.get_models_for_preset(preset_id)

        weights_path = self.registry.get_weights_path(preset_id)
        if os.path.exists(weights_path):
            spatial_model.load_weights(weights_path)

        if inference_region:
            region = SlideRegion.from_dict(inference_region)
            coords = iter_patch_coords(region, patch_size, max_patches=1)
            x, y = coords[0]

        if not regional_spots:
            regional_spots = [
                RegionalSpatialSpot(
                    spot_id="center_spot",
                    x_local=float(patch_size / 2),
                    y_local=float(patch_size / 2),
                    x_wsi=float(x + patch_size / 2),
                    y_wsi=float(y + patch_size / 2),
                    target_expressions=None,
                )
            ]

        registry, patch_region, patch_id = self._process_patch(
            slide_data, slide_id, preset_id, x, y, patch_size
        )

        if registry is None:
            return {
                "status": "empty",
                "message": "В выбранной области нет сегментированных клеток.",
            }

        prediction_output: RegionalPredictionOutput = (
            spatial_model.predict_spatial_expression(
                patch_registry=registry,
                regional_spots=regional_spots,
            )
        )

        roi_coordinates = {
            "x": x,
            "y": y,
            "patch_size": patch_size,
            "preset_id": preset_id,
        }
        if inference_region:
            roi_coordinates["inference_region"] = SlideRegion.from_dict(
                inference_region
            ).to_dict()

        predicted_data_dict = (
            prediction_output.model_dump()
            if hasattr(prediction_output, "model_dump")
            else prediction_output.dict()
        )

        interpretation_data = None
        if include_interpretation and patch_region is not None:
            interpretations = self.interpretation_service.interpret_patch(
                preset_id,
                patch_image=self._patch_tensor(patch_region),
                cell_registry=registry,
                target_gene=target_gene,
                patch_id=patch_id,
            )
            interpretation_data = [
                item.model_dump() if hasattr(item, "model_dump") else item.dict()
                for item in interpretations
            ]
            predicted_data_dict["interpretation"] = interpretation_data

        db_report = self.db.create_analysis_report(
            user_id=user_id,
            slide_id=slide_id,
            preset_id=preset_id,
            roi_coordinates=roi_coordinates,
            predicted_data=predicted_data_dict,
        )

        self.db.log_action(
            user_id=user_id,
            action=f"RUN_INFERENCE | Preset: {preset_id} | Report: {db_report.id}",
            ip_address=ip_address,
        )

        return {
            "status": "success",
            "report_id": db_report.id,
            "preset_id": preset_id,
            "created_at": db_report.created_at.isoformat(),
            "data": predicted_data_dict,
            "interpretation": interpretation_data,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _collect_training_registries(
        self,
        slide_data: Dict[str, Any],
        slide_id: str,
        preset_id: str,
        training_region: SlideRegion,
        patch_size: int,
        max_patches: int,
    ) -> List[WSICellFeatureRegistry]:
        registries: List[WSICellFeatureRegistry] = []
        coords = iter_patch_coords(training_region, patch_size, max_patches=max_patches)

        for x, y in coords:
            registry, _, _ = self._process_patch(
                slide_data, slide_id, preset_id, x, y, patch_size
            )
            if registry and registry.cells:
                registries.append(registry)

        logger.info(f"Found {len(registries)} registries.")
        return registries

    def start_background_training(
        self,
        user_id: str,
        slide_id: str,
        epochs: int,
        ip_address: Optional[str] = None,
        preset_id: Optional[str] = None,
        training_region: Optional[Dict] = None,
        patch_size: int = 1024,
        max_patches: int = 20,
    ) -> str:
        preset_id = self._resolve_preset_id(preset_id)
        slide_data = self._ensure_slide_is_cached(slide_id)
        metadata: WSIContainer = slide_data["metadata"]

        if training_region:
            region = SlideRegion.from_dict(training_region)
        else:
            w, h = metadata.dimensions_level_0
            region = SlideRegion(x=0, y=0, width=w, height=h)

        job_id = f"job_{uuid.uuid4().hex[:8]}"
        self.jobs_registry[job_id] = {
            "status": "running",
            "epoch": 0,
            "loss": 0.0,
            "total_epochs": epochs,
            "preset_id": preset_id,
            "slide_id": slide_id,
            "training_region": region.to_dict(),
            "max_patches": max_patches,
        }

        self.db.update_slide_status(slide_id, status="processing")
        self.db.log_action(
            user_id=user_id,
            action=f"START_TRAINING | Job: {job_id} | Preset: {preset_id} | Slide: {slide_id}",
            ip_address=ip_address,
        )

        def training_worker():
            try:
                _, spatial_model = self.registry.get_models_for_preset(preset_id)
                preset = self.registry.get_preset(preset_id)
                if preset.composite_expert_key:
                    expert_key = preset.composite_expert_key
                else:
                    expert_model, _ = self.registry.get_models_for_preset(preset_id)
                    expert_key = getattr(
                        expert_model, "EXPERT_KEY", preset.expert_ids[0]
                    )

                registries = self._collect_training_registries(
                    slide_data,
                    slide_id,
                    preset_id,
                    region,
                    patch_size,
                    max_patches,
                )

                if not registries:
                    raise ValueError("Не найдено патчей с клетками в зоне обучения")

                gene_names = slide_data["gene_loader"].get_all_gene_names()
                target_genes = (
                    list(getattr(spatial_model, "gene_names", [])) or gene_names[:50]
                )

                dataset = XeniumGraphDataset(
                    registries=registries,
                    target_gene_names=target_genes,
                    all_gene_names=gene_names,
                    expert_key=expert_key,
                )

                if len(dataset) == 0:
                    raise ValueError("Датасет пуст после сборки графов")

                save_dir = os.path.dirname(self.registry.get_weights_path(preset_id))
                model_name = preset.spatial_model_id

                feature_dim = int(dataset[0].x.shape[1])
                if hasattr(spatial_model, "_ensure_model"):
                    spatial_model._ensure_model(feature_dim)
                runtime_model = cast(Any, spatial_model).model
                optimizer = torch.optim.Adam(runtime_model.parameters(), lr=0.001)
                train_pipeline = SpatialTrainPipeline(
                    dataset=dataset,
                    spatial_model=runtime_model,
                    optimizer=optimizer,
                    save_dir=save_dir,
                    model_name=model_name,
                )

                train_pipeline.train(epochs=epochs)
                train_pipeline.load_best_model()
                best_src = os.path.join(save_dir, f"{model_name}_best.pt")
                best_dst = self.registry.get_weights_path(preset_id)
                if os.path.exists(best_src):
                    spatial_model.load_weights(best_src)
                    os.makedirs(os.path.dirname(best_dst), exist_ok=True)
                    spatial_model.save_weights(best_dst)

                self.jobs_registry[job_id]["status"] = "success"
                self.jobs_registry[job_id]["loss"] = train_pipeline.best_loss
                self.jobs_registry[job_id]["graphs_trained"] = len(dataset)
                self.db.update_slide_status(slide_id, status="ready")
                self.db.log_action(
                    user_id=user_id,
                    action=f"TRAINING_SUCCESS | Job {job_id} | Preset {preset_id}",
                    ip_address=ip_address,
                )
            except Exception as e:
                import traceback

                self.jobs_registry[job_id]["status"] = "failed"
                self.jobs_registry[job_id]["error"] = str(e)
                self.jobs_registry[job_id]["traceback"] = traceback.format_exc()
                logger.error(f"Training failed for job {job_id}: {e}", exc_info=True)
                
                self.jobs_registry[job_id]["status"] = "failed"
                self.jobs_registry[job_id]["error"] = str(e)
                self.db.update_slide_status(slide_id, status="failed")
                self.db.log_action(
                    user_id=user_id,
                    action=f"TRAINING_FAILED | Job {job_id}: {str(e)}",
                    ip_address=ip_address,
                )

        threading.Thread(target=training_worker, daemon=True).start()
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self.jobs_registry.get(job_id, {"status": "not_found"})

    def interpret_report(
        self,
        report_id: str,
        target_gene: Optional[str] = None,
    ) -> Dict[str, Any]:
        report = cast("AnalysisReport | None", self.db.get_report_by_id(report_id))
        if not report:
            raise KeyError(f"Report {report_id} not found")

        stored = report.predicted_expression_data or {}
        if "interpretation" in stored:
            return {"report_id": report_id, "interpretation": stored["interpretation"]}

        roi = report.roi_coordinates or {}
        preset_id = roi.get("preset_id", report.preset_id)
        slide_id = str(report.slide_id)
        x = int(roi.get("x", 0))
        y = int(roi.get("y", 0))
        patch_size = int(roi.get("patch_size", 512))

        slide_data = self._ensure_slide_is_cached(slide_id)
        registry, patch_region, patch_id = self._process_patch(
            slide_data, slide_id, preset_id, x, y, patch_size
        )
        if registry is None:
            raise ValueError("Cannot reinterpret empty ROI")

        interpretations = self.interpretation_service.interpret_patch(
            preset_id,
            patch_image=self._patch_tensor(patch_region),
            cell_registry=registry,
            target_gene=target_gene,
            patch_id=patch_id,
        )
        return {
            "report_id": report_id,
            "interpretation": [
                item.model_dump() if hasattr(item, "model_dump") else item.dict()
                for item in interpretations
            ],
        }

    # ------------------------------------------------------------------
    # Users & projects
    # ------------------------------------------------------------------

    def create_new_project(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> str:
        project = cast(
            "Project",
            self.db.create_project(user_id=user_id, name=name, description=description),
        )
        self.db.log_action(
            user_id, f"CREATE_PROJECT | '{name}' (ID: {project.id})", ip_address
        )
        return project.id

    def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        user_projects = cast(List["Project"], self.db.get_user_projects(user_id))
        available_presets = self.registry.get_available_presets()

        projects_dashboard = []
        for proj in user_projects:
            slides = cast(List["XeniumSlide"], self.db.get_project_slides(str(proj.id)))
            projects_dashboard.append(
                {
                    "project_id": proj.id,
                    "name": proj.name,
                    "description": proj.description,
                    "slides_count": len(slides),
                    "slides": [
                        {
                            "id": s.id,
                            "patient": s.patient_name,
                            "status": s.status,
                            "metadata": s.slide_metadata,
                            "reports": [
                                {
                                    "id": report.id,
                                    "created_at": report.created_at.isoformat(),
                                    "preset_id": report.preset_id,
                                    "roi_coordinates": report.roi_coordinates,
                                }
                                for report in self.db.get_slide_reports(str(s.id))
                            ],
                        }
                        for s in slides
                    ],
                }
            )

        return {
            "projects": projects_dashboard,
            "system_presets": [
                {
                    **(p.model_dump() if hasattr(p, "model_dump") else p.dict()),
                    "is_active": p.preset_id == self.registry._active_preset_id,
                }
                for p in available_presets.values()
            ],
        }

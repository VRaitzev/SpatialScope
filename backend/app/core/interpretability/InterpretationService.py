from typing import List, Optional

from app.core.interpretability.AbstractModelInterpreter import InterpretationResult
from app.core.interpretability.EfficientNetInterpreter import EfficientNetInterpreter
from app.core.interpretability.GNNInterpreter import GNNInterpreter
from app.core.ml_models.wsi_models.expert_models.EfficientNetExpertAdapter import EfficientNetExpertAdapter
from app.core.registry.ModelPresetRegistry import ModelPresetRegistry


class InterpretationService:
    """Runs all applicable interpreters for a preset and inference context."""

    def __init__(self, registry: ModelPresetRegistry):
        self.registry = registry

    def interpret_patch(
        self,
        preset_id: str,
        *,
        patch_image=None,
        cell_registry=None,
        target_gene: Optional[str] = None,
        patch_id: str = "patch",
    ) -> List[InterpretationResult]:
        preset = self.registry.get_preset(preset_id)
        _, spatial_model = self.registry.get_models_for_preset(preset_id)
        results: List[InterpretationResult] = []

        for expert_id in preset.expert_ids:
            expert = self.registry.get_expert(expert_id)
            if isinstance(expert, EfficientNetExpertAdapter):
                interpreter = EfficientNetInterpreter(expert)
                results.append(
                    interpreter.interpret(
                        patch_image=patch_image,
                        cell_registry=cell_registry,
                        target_gene=target_gene,
                        patch_id=patch_id,
                    )
                )

        gnn_interpreter = GNNInterpreter(spatial_model)
        results.append(
            gnn_interpreter.interpret(
                cell_registry=cell_registry,
                target_gene=target_gene,
                patch_id=patch_id,
            )
        )
        return results

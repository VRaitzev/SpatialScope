from typing import Optional, List, Tuple, Dict
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
import torch

# =====================================================================
# WSI METADATA SCHEMAS
# =====================================================================

class WSIUploadRequest(BaseModel):
    slide_id: str = Field(..., description="Unique slide ID or patient identifier.")
    source_url: str = Field(..., description="Download URI or local path.")
    staining: str = Field("H&E", description="Staining protocol (e.g., H&E, IHC).")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Lab metadata.")


class WSILevelMetadata(BaseModel):
    level: int = Field(..., description="Pyramid level index (0 = max resolution).")
    dimensions: Tuple[int, int] = Field(..., description="Width and height in pixels.")
    downsample: float = Field(..., description="Downsample factor relative to level 0.")


class WSIContainer(BaseModel):
    slide_id: str = Field(..., description="Unique slide identifier.")
    file_path: str = Field(..., description="Server-side storage path.")
    dimensions_level_0: Tuple[int, int] = Field(..., description="Full-resolution (level 0) dimensions.")
    mpp: float = Field(..., description="Microns per pixel resolution.")
    staining: str = Field("H&E", description="Staining type.")
    levels_count: int = Field(..., description="Total pyramid levels.")
    levels_info: List[WSILevelMetadata] = Field(..., description="Metadata per level.")
    thumbnail_url: Optional[str] = Field(None, description="Preview URI/Base64 string.")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Scanner header info.")

    @model_validator(mode="after")
    def validate_dimensions(self) -> 'WSIContainer':
        if len(self.levels_info) != self.levels_count:
            raise ValueError("levels_info count must match levels_count.")
        return self


# =====================================================================
# PATCH CONTAINERS (DATA LAYER)
# =====================================================================

class WSIRegionContainer(BaseModel):
    """
    [DATA LAYER] Standardized container for tissue patches.
    Ensures PyTorch-ready tensors and global spatial coordinate mapping.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    slide_id: str = Field(..., description="Parent slide identifier.")
    image_tensor: torch.Tensor = Field(..., description="Tensor format: (C, H, W) RGB.")
    
    # Global coordinates (Level 0) for cell-to-WSI mapping
    x_wsi: int = Field(..., description="Global X coordinate of patch top-left corner.")
    y_wsi: int = Field(..., description="Global Y coordinate of patch top-left corner.")
    width_wsi: int = Field(..., description="Patch width in level 0 pixels.")
    height_wsi: int = Field(..., description="Patch height in level 0 pixels.")
    
    extracted_level: int = Field(0, description="Source pyramid level.")
    mpp: float = Field(..., description="Effective microns per pixel at this level.")

    @field_validator('image_tensor', mode='before')
    @classmethod
    def validate_tensor_structure(cls, v: torch.Tensor) -> torch.Tensor:
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(v)}")
        if v.numel() == 0:
            raise ValueError("Tensor is empty.")
        if v.ndim != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got shape {v.shape}")
        if v.shape[0] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {v.shape[0]}")
        if v.dtype != torch.float32:
            raise ValueError(f"Expected torch.float32, got {v.dtype}")
        return v
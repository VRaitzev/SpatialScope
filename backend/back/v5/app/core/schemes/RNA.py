from pydantic import BaseModel, Field
from typing import List

class SpotInPatch(BaseModel):
    """
    [DATA LAYER] Compact representation of a spatial transcriptomic spot 
    localized within a histological patch.
    """
    spot_id: str = Field(..., description="Unique transcriptomic barcode identifier.")
    
    # Local coordinates are essential for spatial pooling within GNN architectures.
    local_x: float = Field(..., description="X-coordinate relative to the top-left corner of the patch.")
    local_y: float = Field(..., description="Y-coordinate relative to the top-left corner of the patch.")
    
    # Gene expression profile
    expression_vector: List[float] = Field(..., description="Vector of normalized gene expression counts.")
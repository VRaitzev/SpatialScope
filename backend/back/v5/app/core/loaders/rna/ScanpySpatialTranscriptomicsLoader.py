import scanpy as sc
import numpy as np
from typing import List, Optional
from app.core.loaders.rna.AbstractTranscriptomicsLoader import AbstractTranscriptomicsLoader
from app.core.schemes.RNA import SpotInPatch

class ScanpySpatialTranscriptomicsLoader(AbstractTranscriptomicsLoader):
    """
    [DATA LAYER] Implementation of spatial transcriptomics data loading using Scanpy.

    This loader provides optimized spatial indexing for AnnData objects, enabling 
    efficient region-based queries of gene expression data for spatial 
    alignment workflows.
    """

    def __init__(self):
        self.adata: Optional[sc.AnnData] = None
        self.coordinates: Optional[np.ndarray] = None  # N x 2 array
        self.expression_matrix: Optional[np.ndarray] = None  # N x M array
        self.spot_ids: List[str] = []

    def load_data(self, file_path: str) -> None:
        """
        Loads the AnnData object and indexes spatial coordinates.

        Args:
            file_path: Path to the .h5ad dataset file.
        """
        self.adata = sc.read_h5ad(file_path)
        
        # Cache spatial coordinates and metadata
        self.coordinates = self.adata.obsm['spatial']
        self.spot_ids = self.adata.obs_names.tolist()
        
        # Convert sparse expression data to dense NumPy array for high-performance slicing
        if hasattr(self.adata.X, "toarray"):
            self.expression_matrix = self.adata.X.toarray()
        else:
            self.expression_matrix = np.array(self.adata.X)

    def get_spots_in_region(
        self, 
        x_wsi: int, 
        y_wsi: int, 
        width_wsi: int, 
        height_wsi: int
    ) -> List[SpotInPatch]:
        """
        Retrieves transcriptomic spots within the specified rectangular histological region.

        Uses NumPy boolean masking for high-performance spatial filtering.

        Args:
            x_wsi: Top-left X coordinate on the WSI coordinate space.
            y_wsi: Top-left Y coordinate on the WSI coordinate space.
            width_wsi: Region width.
            height_wsi: Region height.

        Returns:
            List of SpotInPatch instances with computed local spatial coordinates.
        """
        if self.adata is None:
            raise RuntimeError("Data not initialized. Call load_data() before querying.")

        # Define patch boundaries
        x_max, y_max = x_wsi + width_wsi, y_wsi + height_wsi

        # Boolean masking for spatial filtering
        in_patch_mask = (
            (self.coordinates[:, 0] >= x_wsi) & (self.coordinates[:, 0] <= x_max) &
            (self.coordinates[:, 1] >= y_wsi) & (self.coordinates[:, 1] <= y_max)
        )
        
        spot_indices = np.where(in_patch_mask)[0]
        
        spots_found: List[SpotInPatch] = []
        
        for idx in spot_indices:
            global_x, global_y = self.coordinates[idx]
            
            # Map global WSI coordinates to local patch-relative coordinates
            spots_found.append(SpotInPatch(
                spot_id=self.spot_ids[idx],
                local_x=float(global_x - x_wsi),
                local_y=float(global_y - y_wsi),
                expression_vector=self.expression_matrix[idx].tolist()
            ))
            
        return spots_found

    def get_all_gene_names(self) -> List[str]:
        """Returns the list of gene identifiers indexed in the AnnData object."""
        return self.adata.var_names.tolist()
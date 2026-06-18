from abc import ABC, abstractmethod
from typing import List
from app.core.schemes.RNA import SpotInPatch

class AbstractTranscriptomicsLoader(ABC):
    """
    [DATA LAYER] Interface for spatial transcriptomics data management.

    This abstraction defines the protocol for loading molecular expression data 
    and performing spatial queries to retrieve gene expression profiles 
    corresponding to specific histological regions.
    """

    @abstractmethod
    def load_data(self, file_path: str) -> None:
        """
        Loads and indexes the spatial transcriptomics dataset.

        Args:
            file_path: System path to the expression matrix and coordinate data.
        """
        pass

    @abstractmethod
    def get_spots_in_region(
        self, 
        x_wsi: int, 
        y_wsi: int, 
        width_wsi: int, 
        height_wsi: int
    ) -> List[SpotInPatch]:
        """
        Retrieves transcriptomic spots within a specified histological region.

        This method maps global Whole-Slide Image (WSI) coordinates to local 
        transcriptomic spot locations, enabling spatial alignment between 
        histology and molecular expression data.

        Args:
            x_wsi: Top-left X coordinate on the global WSI.
            y_wsi: Top-left Y coordinate on the global WSI.
            width_wsi: Width of the requested region.
            height_wsi: Height of the requested region.

        Returns:
            A list of SpotInPatch objects containing localized expression data.
        """
        pass

    @abstractmethod
    def get_all_gene_names(self) -> List[str]:
        """
        Retrieves the complete catalog of genes available in the dataset.

        Returns:
            A list of gene identifiers (e.g., Gene Symbols or Ensembl IDs).
        """
        pass
from abc import ABC, abstractmethod
from app.core.schemes.WSI import WSIContainer, WSIRegionContainer

class AbstractWSILoader(ABC):
    """
    [DATA LAYER] Interface for Whole-Slide Image (WSI) data ingestion.

    This abstraction defines the contract for handling high-resolution histological 
    pyramidal images, ensuring memory-efficient access to tissue regions without 
    loading entire slides into RAM.
    """

    @abstractmethod
    def load_slide_metadata(self, file_path: str) -> WSIContainer:
        """
        Parses WSI headers and metadata.

        Extracts critical information (e.g., dimensions, resolution levels) 
        from formats such as .svs, .tiff, or .ndpi and encapsulates it into a 
        lightweight WSIContainer instance.

        Args:
            file_path: System path to the WSI file.

        Returns:
            A WSIContainer instance containing slide metadata.
        """
        pass

    @abstractmethod
    def load_region(
        self, 
        container: WSIContainer, 
        x: int, 
        y: int, 
        width: int, 
        height: int, 
        level: int = 0
    ) -> WSIRegionContainer:
        """
        Extracts a specific rectangular patch from the WSI pyramid.

        This method supports multi-resolution access, enabling efficient 
        retrieval of image context required for CNN-based morphological analysis.

        Args:
            container: The WSIContainer associated with the slide.
            x: Global X coordinate of the region start.
            y: Global Y coordinate of the region start.
            width: Width of the region to extract.
            height: Height of the region to extract.
            level: Pyramidal resolution level (default 0 is highest resolution).

        Returns:
            A WSIRegionContainer containing the extracted image patch.
        """
        pass
import numpy as np
from stardist.models import StarDist2D
from shapely.geometry import Polygon as ShapelyPolygon

from app.core.schemes.Segmentation import (
    SegmentationContainer, 
    SegmentationCellInstance, 
    SegmentationModelManifest, 
    SegmentationFeatureMeta,
    PolygonModel
)
from app.core.schemes.WSI import WSIRegionContainer
from app.core.ml_models.wsi_models.segmentations_models import AbstractSegmentationModel

class StarDistSegmentationAdapter(AbstractSegmentationModel):
    """
    [MODEL LAYER] Wrapper for StarDist 2D cell nuclear segmentation.

    Provides a standardized interface for integrating star-convex object 
    segmentation into the histological analysis pipeline.
    """

    def __init__(self, model_path: str):
        """
        Initializes the StarDist model.

        Args:
            model_path: Identifier or path to the pre-trained StarDist model.
        """
        self.model = StarDist2D.from_pretrained(model_path)
        
    @property
    def manifest(self) -> SegmentationModelManifest:
        """Returns the model manifest with feature definitions."""
        return SegmentationModelManifest(
            model_name="StarDist_Versatile",
            model_version="1.0",
            model_description="StarDist cell nuclear segmentation for histological tissue",
            supported_features=[
                SegmentationFeatureMeta(
                    feature_name="nuclear_area", 
                    description="Planar area of the segmented nucleus", 
                    unit="px²"
                )
            ]
        )

    def predict(self, patch: WSIRegionContainer) -> SegmentationContainer:
        """
        Executes nuclear segmentation on an image patch.

        Args:
            patch: WSIRegionContainer containing the histological image tensor.

        Returns:
            SegmentationContainer populated with detected cellular polygons.
        """
        # Permute CHW tensor to HWC for StarDist processing
        img = patch.image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # StarDist inference
        _, details = self.model.predict_instances(img)
        
        instances = []
        for i in range(len(details['points'])):
            # Convert StarDist coordinates to Pydantic polygon model
            points = [(float(p[1]), float(p[0])) for p in details['coord'][i].T]
            polygon = ShapelyPolygon(points)
            
            instances.append(SegmentationCellInstance(
                polygon=PolygonModel(points=points),
                bbox=tuple(polygon.bounds),  # (minx, miny, maxx, maxy)
                score=float(details['prob'][i])
            ))
            
        return SegmentationContainer(
            instances=instances, 
            image_shape=(img.shape[0], img.shape[1])
        )

    def load_weights(self, path: str):
        """Placeholder for custom weight loading logic."""
        pass

    def save_weights(self, path: str):
        """Placeholder for model weight serialization."""
        pass
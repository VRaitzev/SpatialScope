import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from app.core.loaders.wsi.image.XeniumWSILoader import XeniumWSILoader
from app.core.loaders.rna.XeniumGenePanelLoader import XeniumGenePanelLoader
from app.core.loaders.wsi.segmentatioin.XeniumCellBoundaryAdapter import (
    XeniumCellBoundaryAdapter
)
from app.core.ml_models.wsi_models.registration_models.SpatialRegistrationService import (
    SpatialRegistrationService
)


def demo_registered_xenium():

    BASE_SIZE = 1024

    data_dir = Path(
        r"D:\PROJECTS\Personal\Study\workspace\diploma\data\full_dataset"
    )

    cells_path = str(data_dir / "cells.parquet")
    matrix_path = str(data_dir / "cell_feature_matrix.h5")

    dapi_image_path = str(data_dir / "morphology.ome.tif")
    he_image_path = str(
        data_dir /
        "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_he_image.ome.tif"
    )

    cache_directory = str(data_dir / "registration_cache")

    print("INIT REGISTRATION")

    reg_service = SpatialRegistrationService(
        dapi_path=dapi_image_path,
        he_path=he_image_path,
        cache_dir=cache_directory,
        target_max_dim=4000,
        mpp_xenium=0.2125
    )

    loader = XeniumWSILoader(cells_path)
    gene_loader = XeniumGenePanelLoader(matrix_path)

    adapter = XeniumCellBoundaryAdapter(
        loader,
        gene_loader
    )

    loader.apply_registration_to_dataframe(
        reg_service
    )

    container = loader.load_slide_metadata(
        he_image_path
    )

    w, h = container.dimensions_level_0

    x_start = int(w * 0.45)
    y_start = int(h * 0.45)

    print(
        f"\nLOAD PATCH x={x_start}, y={y_start}"
    )

    region = loader.load_region(
        container,
        x=x_start,
        y=y_start,
        width=BASE_SIZE,
        height=BASE_SIZE
    )

    seg_container = adapter.load_boundaries(
        region,
        registration_service=reg_service
    )

    patch_viz = (
        region.image_tensor
        .permute(1, 2, 0)
        .numpy()
    )

    if patch_viz.max() > 1:
        patch_viz = (
            np.clip(
                patch_viz,
                0,
                255
            )
            .astype(np.uint8)
        )

    print("\nPATCH INFO")
    print("image:", patch_viz.shape)

    print(
        "region:",
        region.x_wsi,
        region.y_wsi,
        region.width_wsi,
        region.height_wsi
    )

    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    ax.imshow(
        patch_viz
    )

    inside = 0
    outside = 0

    centers_x = []
    centers_y = []

    for i, cell in enumerate(
        seg_container.instances
    ):

        if not cell.polygon:
            continue

        poly_points = [

            (
                p[0] - region.x_wsi,
                p[1] - region.y_wsi

            )

            for p in cell.polygon.points
        ]

        poly_np = np.array(
            poly_points
        )

        visible = (

            poly_np[:, 0].max() >= 0
            and
            poly_np[:, 0].min() <= BASE_SIZE

            and

            poly_np[:, 1].max() >= 0
            and
            poly_np[:, 1].min() <= BASE_SIZE

        )

        if not visible:
            outside += 1
            continue

        inside += 1

        polygon = patches.Polygon(

            poly_points,

            closed=True,

            edgecolor="lime",

            facecolor="none",

            linewidth=1,

            alpha=0.85
        )

        ax.add_patch(
            polygon
        )

        if cell.centroid:

            centers_x.append(
                cell.centroid[0]
                - region.x_wsi
            )

            centers_y.append(
                cell.centroid[1]
                - region.y_wsi
            )

        if i == 0:

            print(
                "\nFIRST POLYGON"
            )

            print(
                poly_points[:5]
            )

    print()

    print(
        "VISIBLE:",
        inside
    )

    print(
        "OUTSIDE:",
        outside
    )

    print(
        "TOTAL:",
        len(
            seg_container.instances
        )
    )

    ax.scatter(
        centers_x,
        centers_y,
        s=1.5,
        color="red",
        alpha=0.7,
        label="Centroids"
    )

    ax.legend()

    plt.title(
        f"Visible {inside}"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    demo_registered_xenium()
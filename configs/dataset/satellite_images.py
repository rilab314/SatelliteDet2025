import os
workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

params = dict(
    dataset=dict(
        module_name="dataset.satellite_images",
        class_name="SatelliteImagesDataset",
        path=os.path.join(workspace_path, "dataset/satellite_images"),
        num_classes=11,
        image_height=384,
        image_width=384,)
)
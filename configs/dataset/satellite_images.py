import os
workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

params = dict(
    dataset=dict(
        module_name="dataset.satellite_lane.satellite_images",
        class_name="SatelliteImagesDataset",
        path=os.path.join(workspace_path, "dataset/satellite_lane"),
        num_classes=12,
        image_height=768,
        image_width=768,
        labels=[
            dict(category_id='000', id=0, priority=11, name='ignore', color=(0, 0, 0)),
            dict(category_id='501', id=1, priority=10, name='center_line', color=(77, 77, 255)),
            dict(category_id='502', id=2, priority=6, name='u_turn_zone_line', color=(77, 178, 255)),
            dict(category_id='503', id=3, priority=7, name='lane_line', color=(77, 255, 77)),
            dict(category_id='504', id=4, priority=3, name='bus_only_lane', color=(255, 153, 77)),
            dict(category_id='505', id=5, priority=8, name='edge_line', color=(255, 77, 77)),
            dict(category_id='506', id=6, priority=4, name='path_change_restriction_line', color=(178, 77, 255)),
            dict(category_id='515', id=7, priority=5, name='no_parking_stopping_line', color=(77, 255, 178)),
            dict(category_id='525', id=8, priority=9, name='guiding_line', color=(255, 178, 77)),
            dict(category_id='530', id=9, priority=0, name='stop_line', color=(77, 102, 255)),
            dict(category_id='531', id=10, priority=1, name='safety_zone', color=(255, 77, 128)),
            dict(category_id='535', id=11, priority=2, name='bicycle_lane', color=(128, 255, 77))
        ]
    )
)
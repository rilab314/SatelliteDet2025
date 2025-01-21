import os
workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

params = dict(
    dataset=dict(
        module_name="datasets.soccer_players",
        class_name="SoccerPlayersDataset",
        path=os.path.join(workspace_path, "dataset/soccer-players"),
        num_classes=20,
        image_height=384,
        image_width=384,)
)

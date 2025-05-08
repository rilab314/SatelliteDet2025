import os
workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 데이터셋 설정 파일 불러오기
_base_ = ['dataset/satellite_images.py']

params = dict(
    training=dict(
        lr=0.00002,
        lr_backbone_names=['backbone.0'],
        lr_backbone=0.00002,
        lr_linear_proj_names=['reference_points', 'sampling_offsets'],
        lr_linear_proj_mult=0.1,
        batch_size=2,
        weight_decay=0.0001,
        epochs=20,  # 위성 이미지에 맞게 조정
        lr_drop=16,
        lr_drop_epochs=None,
        clip_max_norm=0.1,
        sgd=False,
        num_workers=2
    ),
    dataset=dict(
        train=dict(
            augmentation=True,
            coco_ann_file='satellite_train_annotations.json'),
        val=dict(
            augmentation=False,
            coco_ann_file='satellite_val_annotations.json'),
        test=dict(augmentation=False),
        augmentation=dict(
            horizontal_flip=dict(p=0.5),
            vertical_flip=dict(p=0.5),  # 위성 이미지에 유용한 증강
            random_resized_crop=dict(
                size=[512, 512],
                scale=[0.5, 1.0],
                ratio=[0.75, 1.333],
                p=1.0),
            # 기타 증강 설정...
        )
    ),
    # 모델 설정
    lightning_model=dict(
        module_name='model.lightning_detr',
        class_name='LitDeformableDETR'
    ),
    core_model=dict(
        module_name='model.deformable_detr',
        class_name='DeformableDETR',
    ),
    # 기타 설정...
    runtime=dict(
        output_dir=os.path.join(workspace_path, 'tblog_satellite'),
        logger_name='satellite_detr',
        device='cuda',
        seed=42,
        resume='',
        start_epoch=0,
        eval=False,
        cache_mode=False
    ),
)
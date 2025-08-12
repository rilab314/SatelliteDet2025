import os
workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

_base_ = ['dataset/satellite_images.py']

params = dict(
    training=dict(
        lr=0.00002,
        lr_backbone_names=['backbone.0'],
        lr_backbone=0.00002,
        lr_linear_proj_names=['reference_points', 'sampling_offsets'], # 삭제 -> 오류 발생하는 것 확인할 것
        lr_linear_proj_mult=0.1,
        batch_size=2,
        weight_decay=0.0001,
        epochs=20,
        lr_drop=8,
        lr_drop_epochs=None,
        clip_max_norm=0.1,
        sgd=False,
        num_workers=2
    ),
    dataset=dict(
        train=dict(augmentation=True,),
        val=dict(augmentation=False,),
        test=dict(augmentation=False),
        augmentation=dict(
            random_brightness_contrast=dict(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5),
            hue_saturation_value=dict(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5),
            random_gamma=dict(
                gamma_limit=[80, 120],
                p=0.5),
            gauss_noise=dict(
                std_range=[0.02, 0.05],
                mean_range=[0.0, 0.0],
                p=0.5))
    ),
    lightning_model=dict(
        module_name='model.lightning_detr',
        class_name='LitDeformableDETR'
    ),
    core_model=dict(
        module_name='model.defm_lanedet',
        class_name='DefmLaneDetector',
    ),
    backbone=dict(
        module_name='model.backbone',
        class_name=['ResNet50_Clip', 'SwinV2_384', 'SwinV2_768'][2],
        output_layers=['layer1', 'layer2', 'layer3', 'layer4'],
        dilation=False,
        position_embedding=dict(
            type='sine',
            scale=6.283185307179586
        ),
    ),
    transformer=dict(
        module_name='model.transformer_enc_only',
        class_name='DeformableTransformerEncoderOnly',
        hidden_dim=256,
        nheads=8,
        enc_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        num_feature_levels=4,
        enc_n_points=4,
        with_box_refine=False,
        aux_loss=True,
        two_stage=True,
        segmentation=False,
        frozen_weights=False
    ),
    postprocessors=dict( # TODO: 사용 X
        line=dict(
            module_name='model.instance_generator',
            class_name='LineStringInstanceGenerator', 
            topk=100,
            score_threshold=0.05),
    ),
    matcher=dict(
        module_name='model.matcher',
        class_name='PointMatcher',
        point_cost=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        cost_headtail=[1.0, 1.0],
        class_cost=2.0
    ),
    criterion=dict(
        module_name='model.criterion',
        class_name='SegmentationCriterion',
    ),
    losses=dict(
        cls_loss=1,
        end_loss=1,
        point_loss=10,
        focal_alpha=0.25
    ),
    evaluation=dict(
        topk=100,
        score_thresh=0.1
    ),
    runtime=dict(
        output_dir=os.path.join(workspace_path, 'tblog'),
        logger_name='defm_detr',
        device='cuda',
        seed=42,
        resume='',
        start_epoch=0,
        eval=False,
        cache_mode=False
    ),
)
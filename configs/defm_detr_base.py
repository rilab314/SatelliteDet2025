import os
workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

_base_ = ['dataset/soccer_players.py']

params = dict(
    training=dict(
        lr=0.00002,
        lr_backbone_names=['backbone.0'],
        lr_backbone=0.00002,
        lr_linear_proj_names=['reference_points', 'sampling_offsets'],
        lr_linear_proj_mult=0.1,
        batch_size=2,
        weight_decay=0.0001,
        epochs=10,
        lr_drop=8,
        lr_drop_epochs=None,
        clip_max_norm=0.1,
        sgd=False,
        num_workers=2),

    dataset=dict(
        train=dict(
            augmentation=True,
            coco_ann_file='instances_train.json'),
        val=dict(
            augmentation=False,
            coco_ann_file='instances_val.json'),
        test=dict(augmentation=False),
        augmentation=dict(
            horizontal_flip=dict(p=0.5),
            random_resized_crop=dict(
                size=[384, 384],
                scale=[0.5, 1.0],
                ratio=[0.75, 1.333],
                p=1.0),
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
                p=0.5))),

    lightning_model=dict(
        module_name='model.lightning_detr',
        class_name='LitDeformableDETR'),

    core_model=dict(
        module_name='model.deformable_detr',
        class_name='DeformableDETR'),

    backbone=dict(
        module_name='model.backbone',
        class_name=['ResNet50_Clip', 'SwinV2_384'][1],
        output_layers=['layer2', 'layer3', 'layer4'],
        dilation=False,
        position_embedding=dict(
            type='sine',
            scale=6.283185307179586)),  # 2 * pi

    transformer=dict(
        module_name='model.deformable_transformer',
        class_name='DeformableTransformer',
        enc_layers=6,
        dec_layers=6,
        num_feature_levels=4,
        dim_feedforward=1024,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=300,
        dec_n_points=4,
        enc_n_points=4,
        return_intermediate_dec=True,
        with_box_refine=False,
        aux_loss=True,
        two_stage=True,
        segmentation=False,
        frozen_weights=False),

    matcher=dict(
        module_name='model.matcher',
        class_name='HungarianMatcher',
        class_cost=2,  # TODO check
        bbox_cost=5,
        giou_cost=2),

    postprocessors=dict(
        bbox=dict(
            module_name='model.postprocess',
            class_name='BoxPostProcess', 
            topk=100,
            score_threshold=0.05),),

    criterion=dict(
        module_name='model.criterion',
        class_name='SetCriterion',),

    losses=dict(
        cls_loss=2,
        bbox_loss=5,
        giou_loss=2,
        mask_loss=1,
        dice_loss=1,
        cardinality=True,
        accuracy=True,
        focal_alpha=0.25),

    evaluation=dict(
        topk=100,
        score_thresh=0.1),

    runtime=dict(
        output_dir=os.path.join(workspace_path, 'tblog'),
        logger_name='defm_detr',
        device='cuda',
        seed=42,
        resume='',
        start_epoch=0,
        eval=False,
        cache_mode=False)
)

import timm
import torch.nn as nn
import settings
from util.print_util import print_data, print_model


def view_models():
    model_names = timm.list_models('resnet50*', pretrained=True)
    print(f'model_names: {model_names}\n')
    model = timm.create_model('resnet50_clip.cc12m', pretrained=True)
    print_data(model.default_cfg, indent=2, title='----- model.default_cfg -----')
    print('\n----- model structure -----')
    print_model(model)

    model_names = timm.list_models('swin*', pretrained=True)
    print(f'\nmodel_names: {model_names}\n')
    model = timm.create_model('swin_base_patch4_window12_384.ms_in22k', pretrained=True)
    print_data(model.default_cfg, indent=2, title='model.default_cfg')
    print('\n----- model structure -----')
    print_model(model)


if __name__ == "__main__":
    view_models()

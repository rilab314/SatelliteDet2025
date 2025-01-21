import numpy as np
import torch.nn as nn
import torch

from util.misc import NestedTensor


def print_data(data, indent=2, level=0, title=None):
    if title is not None:
        print(f"<------ {title}")
    print_data_inner(data, '', indent, level)
    if title is not None:
        print(f"------>")


def print_data_inner(data, key: str, indent=2, level=0):
    prefix = spaces = ' ' * indent * level  # 들여쓰기 공백
    if key:
        prefix += key + ': '
    if len(str(data)) < 150:
        print(f"{prefix}{data}")
    elif isinstance(data, torch.Tensor):
        print(f"{prefix}tensor{tuple(data.shape)}")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}np{data.shape}")
    elif isinstance(data, NestedTensor):
        print(f"{prefix}NestedTensor [tensor{tuple(data.tensors.shape)}, mask{tuple(data.mask.shape)}]")
    elif isinstance(data, dict):
        print(f"{prefix}{'{'}")
        for new_key, value in data.items():                    
            print_data_inner(value, new_key, indent, level+1)
        print(f"{spaces}{'}'}")
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}{'['}")
        for new_key, value in enumerate(data):
            new_key = str(new_key)
            print_data_inner(value, new_key, indent, level+1)
        print(f"{spaces}{']'}")
    else:
        print(f"{prefix}{str(data)[:100]}")



def print_model(model, max_depth=None):
    """
    모델의 계층적 구조를 번호와 함께 들여쓰기로 출력하는 함수
    모델의 모든 모듈을 탐색하고, 각 계층에 번호를 붙여서 출력합니다.
    """
    def print_layers(module, level, prefix, child_name=None, depth=0):
        if max_depth is not None and depth > max_depth:
            return
        level = level + '.' if level else ''
        if isinstance(module, nn.Module):
            module_name = module.__class__.__name__
            if hasattr(module, 'named_children') and len(list(module.named_children())) > 0:
                if child_name:
                    print(f"{prefix}{level} {module_name} (name: {child_name})")
                else:
                    print(f"{prefix}{level}{module_name}")
            elif module_name == 'Conv2d':
                print(f"{prefix}{level} {module}")
            elif module_name == 'Linear':
                print(f"{prefix}{level} {module}")
            else:
                print(f"{prefix}{level} {module_name}")

            # 해당 레이어의 자식 모듈들이 있다면 재귀적으로 호출
            for i, (child_name, child) in enumerate(module.named_children(), 1):
                # 번호를 붙여서 자식 모듈을 출력
                print_layers(child, f"{level}{i}", f"{prefix}  ", child_name, depth+1)

    print_layers(model, "", "", None, 0)

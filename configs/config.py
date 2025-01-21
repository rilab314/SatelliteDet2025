import os
import sys
import importlib
import copy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


class CfgNode:
    @staticmethod
    def from_file(cfg_name):
        params = load_py_config(cfg_name)
        return CfgNode(**params)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = CfgNode(**value)
            setattr(self, key, value)
    
    def __repr__(self):
        return self._repr_inner(0)
    
    def _repr_inner(self, indent_level):
        # 클래스의 속성들을 계층적으로 표현하기 위한 재귀 함수
        indent = '  ' * indent_level
        lines = []
        for key, value in self.__dict__.items():
            if isinstance(value, CfgNode):
                lines.append(f"{indent}{key}:\n{value._repr_inner(indent_level + 1)}")
            else:
                lines.append(f"{indent}{key}: {value}")
        return '\n'.join(lines)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, CfgNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __contains__(self, key):
        return key in self.__dict__
    
    def __getitem__(self, key):
        if key in self.__dict__:
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in CfgNode.")
    
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = CfgNode(**value)
        setattr(self, key, value)
    
    def get(self, key, default=None):
        if key in self.__dict__:
            return getattr(self, key)
        else:
            return default


def load_py_config(cfg_name):
    """
    `path` 위치의 config.py 모듈을 import하여,
    _base_ 순서대로 병합 후, 최종 params 딕셔너리를 반환한다.
    """
    cfg_name = cfg_name.replace('.py', '').replace('/', '.').strip('.')
    cfg_name = 'configs.' + cfg_name
    mod = importlib.import_module(cfg_name)

    merged_params = {}
    if hasattr(mod, "_base_"):
        base_list = getattr(mod, "_base_")
        if not isinstance(base_list, list):
            raise ValueError(f"_base_ must be a list, got {type(base_list)}")

        for base_file in base_list:
            base_params = load_py_config(base_file)  # 재귀
            merged_params = merge_dicts(merged_params, base_params)

    if not hasattr(mod, "params"):
        raise ValueError(f"No 'params' found in {cfg_name}")
    local_params = getattr(mod, "params")
    if not isinstance(local_params, dict):
        raise ValueError(f"'params' in {cfg_name} must be a dict, got {type(local_params)}")

    merged_params = merge_dicts(merged_params, local_params)
    return merged_params


def merge_dicts(base, new):
    """
    두 dict를 병합하는 함수.
    중첩 딕셔너리도 재귀적으로 병합하고,
    충돌 시 'new' 값이 base를 덮어씀.
    """
    merged = copy.deepcopy(base)
    for k, v in new.items():
        if (k in merged and isinstance(merged[k], dict) and isinstance(v, dict)):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def example():
    # 사용 예시
    cfg = CfgNode.from_file('defm_detr_base')
    print(cfg.dataset)


if __name__ == "__main__":
    example()

import importlib


def build_dataset(cfg, split):
    module = importlib.import_module(cfg.dataset.module_name)
    dataset_class = getattr(module, cfg.dataset.class_name)
    dataset = dataset_class(cfg, split)
    return dataset

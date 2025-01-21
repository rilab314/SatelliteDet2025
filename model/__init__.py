import util.misc as misc


def build_model(cfg):
    model = misc.build_instance(cfg.lightning_model.module_name, cfg.lightning_model.class_name, cfg)
    return model

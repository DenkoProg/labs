from omegaconf import DictConfig
from pytorch_lightning import loggers

from coinin_vlm.utils import dictconfig_to_dict

REGISTRY = {
    "tensorboard": loggers.TensorBoardLogger,
}


def get_logger_by_cfg(cfg: DictConfig):
    builder = REGISTRY[cfg.name]
    kwargs = dictconfig_to_dict(cfg.args)
    return builder(**kwargs)

from typing import Any

from omegaconf import DictConfig, OmegaConf


def dictconfig_to_dict(cfg: DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    ret = {}
    for k in cfg.keys():
        if isinstance(cfg[k], DictConfig) or isinstance(cfg[k], dict):
            ret[k] = dictconfig_to_dict(cfg[k])
        else:
            ret[k] = cfg[k]

    return ret

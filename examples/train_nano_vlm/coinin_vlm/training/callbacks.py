from pathlib import Path

from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint


def callback_from_config(cfg: DictConfig) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=Path.cwd() / "checkpoints",
        monitor=cfg.get("monitor", "val/loss_epoch"),
        mode=cfg.get("mode", "min"),
        save_top_k=1,
        verbose=True,
        filename="best-{epoch}",
        save_last=True,
    )

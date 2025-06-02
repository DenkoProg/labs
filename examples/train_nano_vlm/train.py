from pathlib import Path

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import rootutils

from coinin_vlm.training.callbacks import callback_from_config
from coinin_vlm.training.loggers import get_logger_by_cfg
from coinin_vlm.utils import dictconfig_to_dict
from src.data import get_data_by_cfg

# from coinin_vlm.pipelines.train import training_pipeline
from src.model.vision_language_model import VisionLanguageModel
from src.module import NanoVLMModule
from src.optimizer import init_optimizer
from src.utils import patch_nano_vlm_configs_from_hydra

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def training_pipeline(cfg: DictConfig) -> None:
    train_cfg, vlm_cfg = patch_nano_vlm_configs_from_hydra(cfg)

    # Logger
    logger = get_logger_by_cfg(cfg.logger)

    # Dataset
    loader_train, loader_val, loader_test = get_data_by_cfg(cfg)

    # Model
    model = VisionLanguageModel(vlm_cfg)
    optimizer = init_optimizer(model, **dictconfig_to_dict(cfg.optimizer.args))
    total_optimization_steps = (
        cfg.trainer.max_epochs * len(loader_train) // cfg.trainer.accumulate_grad_batches
    )
    module = NanoVLMModule(
        model,
        optimizer,
        train_cfg.lr_mp,
        train_cfg.lr_backbones,
        total_optimization_steps=total_optimization_steps,
    )

    # Trainer
    trainer_kwargs = dictconfig_to_dict(cfg.trainer)
    trainer = pl.Trainer(
        default_root_dir=str(Path.cwd()),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        callbacks=[callback_from_config(cfg.callback)],
        logger=logger,
        **trainer_kwargs,
    )
    # We save on best validation loss, measure test acuracy after
    # Originally, model was saved on best test accuracy after each epoch
    trainer.fit(model=module, train_dataloaders=loader_train, val_dataloaders=loader_val)
    if loader_test is not None:
        trainer.test(model=module, dataloaders=loader_test)


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.3")
def main_hydra(cfg: DictConfig) -> None:
    training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()

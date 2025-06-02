from omegaconf import DictConfig

from src.model import config


def patch_nano_vlm_configs_from_hydra(
    cfg: DictConfig,
) -> tuple[config.VLMConfig, config.TrainConfig]:
    """
    Patch the NanoVLM configuration from Hydra's DictConfig.
    This function modifies the configuration in place.
    """
    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    # Data
    # train_cfg.val_ratio = cfg.data.val.set.ratio
    # train_cfg.batch_size = cfg.data.train.loader.batch_size
    # train_cfg.mmstar_batch_size = cfg.data.test.loader.batch_size

    # Model
    train_cfg.compile = cfg.compile
    if weights_p := cfg.model.args.weights:
        print(f"Loading VLM weights from {weights_p}")
        vlm_cfg.vlm_checkpoint_path = weights_p
    if cfg.resume_from_checkpoint:
        assert vlm_cfg.vlm_checkpoint_path is not None, (
            "Please provide a VLM checkpoint path to resume from."
        )
        train_cfg.resume_from_vlm_checkpoint = True
        vlm_cfg.vlm_load_backbone_weights = False

    # Optimizer
    train_cfg.lr_mp = cfg.optimizer.args.lr_mp
    train_cfg.lr_backbones = cfg.optimizer.args.lr_backbones

    # Training
    train_cfg.epochs = cfg.trainer.max_epochs
    train_cfg.gradient_accumulation_steps = cfg.trainer.accumulate_grad_batches

    return train_cfg, vlm_cfg

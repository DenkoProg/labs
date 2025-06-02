import torch

from src.model.vision_language_model import VisionLanguageModel


def init_optimizer(
    model: VisionLanguageModel,
    # train_cfg,
    lr_mp: float = 2e-3,
    lr_backbones: float = 1e-4,
) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": model.MP.parameters(),
            "lr": lr_mp,
        },
        {
            "params": list(model.decoder.parameters()) + list(model.vision_encoder.parameters()),
            "lr": lr_backbones,
        },
    ]
    return torch.optim.AdamW(param_groups)

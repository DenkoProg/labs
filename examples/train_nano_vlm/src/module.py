from collections.abc import Callable
import math
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch

from src.model.utils import check_multiple_choice_with_regex
from src.model.vision_language_model import VisionLanguageModel
from src.optimizer import init_optimizer
from src.tokenizer import get_tokenizer


def apply_lr_schedule(it: int, max_lr: float, max_steps: int) -> float:
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


class NanoVLMModule(pl.LightningModule):
    def __init__(
        self,
        model: VisionLanguageModel,
        optimizer: torch.optim.Optimizer | None = None,
        lr_mp: float | None = None,
        lr_backbones: float | None = None,
        total_optimization_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer

        # Scheduling parameters
        self.apply_scheduling = None not in (lr_mp, lr_backbones, total_optimization_steps)
        self.lr_mp = lr_mp
        self.lr_backbones = lr_backbones

        self.optimization_step = 0
        self.total_optimization_steps = total_optimization_steps

        # Cache for storing intermediate results
        self.cache = {}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = init_optimizer(self.model)
        return self.optimizer

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        # Scheduling from nanoVLM training script
        self.optimization_step += 1
        if self.apply_scheduling:
            lr_mp = apply_lr_schedule(
                self.optimization_step, self.lr_mp, self.total_optimization_steps
            )
            lr_backbones = apply_lr_schedule(
                self.optimization_step, self.lr_backbones, self.total_optimization_steps
            )
            optimizer.param_groups[0]["lr"] = lr_mp
            optimizer.param_groups[1]["lr"] = lr_backbones

            # Log
            for name, value in [["lr_mp", lr_mp], ["lr_backbones", lr_backbones]]:
                self.log(name, value, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure=optimizer_closure,
        )

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        # Unpack
        images = batch["image"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Inference
        return self.model(input_ids, images, attention_mask=attention_mask, targets=labels)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        _, loss = self(batch)
        self.log(
            "train/loss",
            loss.item(),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            # batch_size=len(batch["image"]),
        )
        return {"loss": loss}

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        with torch.no_grad():
            _, loss = self(batch)
        self.log(
            "val/loss",
            loss.item(),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            # batch_size=len(batch["image"]),
        )
        return {"loss": loss}

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        tokenizer = get_tokenizer(self.model.cfg.lm_tokenizer)

        # Unpack
        image = batch["images"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Inference
        correct_answer = tokenizer.batch_decode(labels, skip_special_tokens=True)
        gen = self.model.generate(input_ids, image, attention_mask)
        model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)
        is_correct = check_multiple_choice_with_regex(model_output, correct_answer)

        # Accumulate in cache
        self.cache["test_total_examples"] += len(is_correct)
        self.cache["test_correct_predictions"] += sum(is_correct)

    def on_test_epoch_start(self):
        self.cache["test_total_examples"] = 0
        self.cache["test_correct_predictions"] = 0
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        n_total = self.cache["test_total_examples"]
        n_correct = self.cache["test_correct_predictions"]
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        self.log(
            "test/accuracy",
            accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return super().on_test_epoch_end()

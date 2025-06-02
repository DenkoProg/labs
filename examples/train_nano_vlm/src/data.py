import random

from datasets import concatenate_datasets, load_dataset
import numpy
from omegaconf import DictConfig

# import wandb
import torch
from torch.utils.data import DataLoader, DistributedSampler

from coinin_vlm.data.datasets.registry import get_dataset_by_cfg
from coinin_vlm.data.wrappers.nano_vlm import NanoVLMDatasetWrapper
from src.dataset.collators import MMStarCollator, VQACollator
from src.dataset.datasets import MMStarDataset, VQADataset
import src.dist_utils as dist_utils
from src.model import config
from src.tokenizer import get_tokenizer
from src.transforms import get_image_transforms
from src.utils import patch_nano_vlm_configs_from_hydra


def get_data_by_cfg(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    # Original
    train_cfg, vlm_cfg = patch_nano_vlm_configs_from_hydra(cfg)
    if cfg.data.get("name", None) == "the_cauldron":
        print(
            "Using original dataset. "
            "Look into pipelines.train_nano_vlm.src.model.config.TrainConfig for more options"
        )
        # Additional configuration for original datasets
        train_cfg.val_ratio = cfg.data.val.set.ratio
        train_cfg.batch_size = cfg.data.train.loader.batch_size
        train_cfg.mmstar_batch_size = cfg.data.test.loader.batch_size
        return get_data_original(train_cfg, vlm_cfg)

    # New (CoinIn)
    loader_train, loaders_val = get_data_coinin(cfg.data, vlm_cfg)
    return loader_train, loaders_val, None  # No test loader for now


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_original(
    train_cfg: config.TrainConfig,
    vlm_cfg: config.VLMConfig,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    # Create datasets
    image_processor = get_image_transforms(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)

    # Load and combine all training datasets
    combined_train_data = []
    for dataset_name in train_cfg.train_dataset_name:
        train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name)
        combined_train_data.append(train_ds["train"])
    train_ds = concatenate_datasets(combined_train_data)

    test_ds = load_dataset(train_cfg.test_dataset_path)
    train_ds = train_ds.shuffle(
        seed=0
    )  # Shuffle the training dataset, so train and val get equal contributions from all concatinated datasets

    # Apply cutoff if specified
    if train_cfg.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    train_dataset = VQADataset(train_ds.select(range(train_size)), tokenizer, image_processor)
    val_dataset = VQADataset(
        train_ds.select(range(train_size, total_samples)), tokenizer, image_processor
    )
    test_dataset = MMStarDataset(test_ds["val"], tokenizer, image_processor)

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders
    train_sampler = DistributedSampler(
        train_dataset,
        rank=dist_utils.get_rank(),
        num_replicas=dist_utils.get_world_size(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,  # =per device BS in DDP
        sampler=train_sampler,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=dist_utils.get_rank(),
        num_replicas=dist_utils.get_world_size(),
        shuffle=False,  # Usually False for validation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.mmstar_batch_size,
        shuffle=False,
        collate_fn=mmstar_collator,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, test_loader


def get_data_coinin(
    cfg: DictConfig,
    vlm_cfg: config.VLMConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = get_dataset_by_cfg(cfg.train.set)
    val_dataset = get_dataset_by_cfg(cfg.val.set)

    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    image_transforms = get_image_transforms(vlm_cfg.vit_img_size)

    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    g = torch.Generator()
    g.manual_seed(0)

    # Train
    train_dataset = NanoVLMDatasetWrapper(
        train_dataset,
        eos_token=tokenizer.eos_token,
        image_transforms=image_transforms,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        rank=dist_utils.get_rank(),
        num_replicas=dist_utils.get_world_size(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.loader.batch_size,  # =per device BS in DDP
        sampler=train_sampler,
        collate_fn=vqa_collator,
        num_workers=cfg.train.loader.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Validation
    val_dataset = NanoVLMDatasetWrapper(
        val_dataset,
        eos_token=tokenizer.eos_token,
        image_transforms=image_transforms,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        rank=dist_utils.get_rank(),
        num_replicas=dist_utils.get_world_size(),
        shuffle=False,  # Usually False for validation
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val.loader.batch_size,
        sampler=val_sampler,
        collate_fn=vqa_collator,
        num_workers=cfg.val.loader.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader

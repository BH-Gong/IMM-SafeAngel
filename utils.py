import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

def get_optimizer(model, cfg):
    opt_name = cfg["optimizer"]

    if opt_name == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0)
        )

    elif opt_name == 'AdamW':
        return optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0)
        )

    elif opt_name == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=0.9,
            weight_decay=cfg.get("weight_decay", 0)
        )

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def get_scheduler(optimizer, cfg):
    name = cfg["scheduler"]

    if name == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["T_max"],
            eta_min=cfg["min_lr"]
        )

    elif name == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg["min_lr"]
        )

    elif name == 'StepLR':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["epochs"] // 3,
            gamma=0.5
        )

    elif name == 'OneCycleLR':
        return "ONE_CYCLE"

    else:
        return None

def get_criterion(cfg, class_weights=None):
    name = cfg["criterion"]

    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(weight=class_weights)

    elif name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=class_weight)

    else:
        raise ValueError(f"Unknown loss: {name}")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from BEATs import BEATs, BEATsConfig
from CNN_base import CNNBaseline
from Transformer_base import TransformerBaseline
from classifier import BEATsClassifier
from utils import get_optimizer, get_scheduler, get_criterion
from datasets3 import BEATsDataset
from train import Trainer
from inference import Inferencer

import time
import os
import json

def load_beats(path, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = BEATsConfig(checkpoint["cfg"])
    
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model

def collate_fn(batch):
    wavs, labels = [], []

    wavs = [item["waveform"] for item in batch]
    labels = [item["target"] for item in batch]

    wavs = torch.stack(wavs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "waveform": wavs,
        "target": labels
    }

def train_dataloader_builder(dataset, batch_size=256, num_workers=4):
    train_sampler = DistributedSampler(dataset, shuffle=True)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, train_sampler

def valid_dataloader_builder(dataset, batch_size=256, num_workers=4):
    valid_sampler = DistributedSampler(dataset, shuffle=False)

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return valid_loader, valid_sampler

def test_dataloader_builder( dataset, batch_size=256, num_workers=4):
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return test_loader

def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank
    
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    with open('./config.json') as json_file:
        config = json.load(json_file)

    mode = config.get('mode')
    train_file = config.get('train_file_list')
    valid_file = config.get('valid_file_list')
    test_file = config.get('test_file_list')
    model_pt_path = config.get('model_pt_path')
    epochs = config.get('epochs')
    save_term = config.get('save_term')
    model_type = config.get('model_type')
    
    beats_model = load_beats(model_pt_path, device=device)

    if mode == 'train':
        train_dataset = BEATsDataset(train_file, sample_rate=16000, chunk_sec=1, mode="train")
        train_loader, train_sampler  = train_dataloader_builder(train_dataset)

        label_map = train_dataset.label_map
        
        valid_dataset = BEATsDataset(valid_file, sample_rate=16000, chunk_sec=1, mode="valid")
        val_loader, valid_sampler = valid_dataloader_builder(valid_dataset)

        if model_type == 'CNN':
            model = CNNBaseline(num_classes=len(label_map)).to(device)
        elif model_type == 'BEATs':
            model = BEATsClassifier(beats_model=beats_model, num_classes=len(label_map)).to(device)
        elif model_type == 'Transformer':
            model = TransformerBaseline(num_classes=len(label_map), d_model=128, nhead=4, num_layers=3,
                                        dim_feedforward=256, dropout=0.1).to(device)
        else:
            raise ValueError
            
        model = DDP(model, device_ids=[local_rank], 
                    output_device=local_rank, find_unused_parameters=True)

        print(model)
        
        num_classes = len(label_map)

        class_counts = torch.tensor(train_dataset.class_counts, dtype=torch.float32)
        
        class_weights = 1.0 / torch.sqrt(class_counts)
        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.to(device)

        optimizer = get_optimizer(model, config)
        criterion = get_criterion(config, class_weights)
        scheduler = get_scheduler(optimizer, config)

        t = Trainer(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                    train_loader=train_loader, val_loader=val_loader,
                    epochs=epochs, save_term=save_term, label_map=label_map, device=device)
        t.trainer()

    elif mode == 'inference':
        # s_time = time.time()

        checkpoint_path = config.get("classifier_pt_path")
        ckpt = torch.load(checkpoint_path, map_location=device)
        label_map = ckpt["label_map"]
        
        if model_type == 'CNN':
            model = CNNBaseline(num_classes=len(label_map)).to(device)
        elif model_type == 'BEATs':
            model = BEATsClassifier(beats_model=beats_model, num_classes=len(label_map)).to(device)
        elif model_type == 'Transformer':
            model = TransformerBaseline(num_classes=len(label_map), d_model=128, nhead=4, num_layers=3,
                                        dim_feedforward=256, dropout=0.1).to(device)
        else:
            raise ValueError
        
        model.load_state_dict(ckpt["model_state_dict"])

        model = DDP(model, device_ids=[local_rank], 
                    output_device=local_rank, find_unused_parameters=True)

        test_dataset = BEATsDataset(test_file, sample_rate=16000, chunk_sec=1, mode="test")
        test_loader = test_dataloader_builder(test_dataset)
        
        inferencer = Inferencer(model=model, test_loader=test_loader, 
                                label_map=label_map, device=device)

        result = inferencer.inferencer()
        
        if len(result) == 3:
            preds, pred_labels, targets = result
            inferencer.evaluate(preds, targets)
        else:
            preds, pred_labels = result
        # e_time = time.time()
        
        # print(e_time - s_time)

if __name__ == "__main__":
    main()
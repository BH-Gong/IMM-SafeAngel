import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import torchaudio
import random
import librosa
import scipy
import os
import json

from tqdm import trange, tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, train_loader, 
                 val_loader, epochs, save_term, label_map, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_term = save_term
        self.scheduler = scheduler
        self.label_map = label_map
        self.history = []
        
    def train_one_epoch(self):
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        disable = dist.is_initialized() and dist.get_rank() != 0
        pbar = tqdm(self.train_loader, total=len(self.train_loader), desc="Training", disable=disable)

        for batch in pbar:
            wav = batch["waveform"].to(self.device, non_blocking=True)
            label = batch["target"].to(self.device, non_blocking=True)
    
            self.optimizer.zero_grad()
            logits = self.model(wav)
            
            loss = self.criterion(logits, label)
            loss.backward()

            self.optimizer.step()

            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if dist.is_initialized():
            loss_tensor = torch.tensor(total_loss, dtype=torch.float64, device=self.device)
            sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=self.device)
    
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    
            global_loss = (loss_tensor / sample_tensor).item()
    
        else:
            global_loss = total_loss / total_samples

        return global_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        disable = dist.is_initialized() and dist.get_rank() != 0
        vpbar = tqdm(self.val_loader, total=len(self.val_loader), desc="Validation", disable=disable)
    
        for batch in vpbar:
            wav = batch["waveform"].to(self.device, non_blocking=True)
            label = batch["target"].to(self.device, non_blocking=True)
    
            logits = self.model(wav)
    
            loss = self.criterion(logits, label)
            batch_size = label.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
        if dist.is_initialized():
            loss_tensor = torch.tensor(total_loss, dtype=torch.float64, device=self.device)
            sample_tensor = torch.tensor(total_samples, dtype=torch.float64, device=self.device)
    
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    
            global_loss = (loss_tensor / sample_tensor).item()
    
        else:
            global_loss = total_loss / total_samples

        return global_loss

    def get_model_state_dict(self, model):
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()

    def trainer(self):
        best_loss = float("inf")
        patience = 10
        patience_cnt = 0

        os.makedirs("./result", exist_ok=True)
        
        disable = dist.is_initialized() and dist.get_rank() != 0
        
        for epoch in trange(self.epochs, disable=disable):
            if (hasattr(self.train_loader, "sampler")
                and isinstance(self.train_loader.sampler, 
                               torch.utils.data.distributed.DistributedSampler)):
                self.train_loader.sampler.set_epoch(epoch)
        
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
        
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(train_loss, val_loss)

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            self.history.append(epoch_log)

            if not dist.is_initialized() or dist.get_rank() == 0:
                with open("./result/loss_history.json", "w") as f:
                    json.dump(self.history, f, indent=4)
        
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
        
            if val_loss < best_loss:
                best_loss = val_loss
                if not dist.is_initialized() or dist.get_rank() == 0:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.get_model_state_dict(self.model),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "label_map": self.label_map,
                    }, "./result/best.pt")

                patience_cnt = 0
            else:
                patience_cnt += 1

            early_stop_flag = torch.tensor(1 if patience_cnt >= patience else 0, device=self.device)
            if dist.is_initialized():
                dist.all_reduce(early_stop_flag, op=dist.ReduceOp.MAX)
                
            if early_stop_flag.item() > 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("Early stopping")
    
                if dist.is_initialized():
                        dist.barrier()
                break

            if epoch == 0 or (epoch + 1) % self.save_term == 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.get_model_state_dict(self.model),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "label_map": self.label_map,
                    }, f"./result/{epoch+1}.pt")
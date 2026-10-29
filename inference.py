import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torchaudio
import random
import librosa
import scipy

from tqdm import trange, tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

class Inferencer:
    def __init__(self, model, test_loader, label_map, device):
        self.model = model
        self.test_loader = test_loader
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.device = device

    @torch.no_grad()
    def inferencer(self):
        all_predictions = []
        all_prediction_labels = []
        all_targets = []

        self.model.eval()
        
        pbar = tqdm(self.test_loader, total=len(self.test_loader), desc="Inference")

        for batch in pbar:
            wav = batch["waveform"].to(self.device)
            logits = self.model(wav).to(self.device)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().tolist())
            all_prediction_labels.extend([self.idx_to_label[idx] for idx in preds.cpu().tolist()])
            
            if "target" in batch:
                if isinstance(batch["target"], torch.Tensor):
                    all_targets.extend(batch["target"].cpu().tolist())
                else:
                    all_targets.extend(batch["target"])

        if all_targets:
            return all_predictions, all_prediction_labels, all_targets
            
        return all_predictions, all_prediction_labels

    def evaluate(self, preds, targets):
        accuracy = accuracy_score(targets, preds) * 100
    
        print("\n" + "=" * 50)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("=" * 50)
    
        target_names = [k for k, v in sorted(self.label_map.items(), key=lambda item: item[1])]
    
        print(classification_report(targets, preds, labels=list(range(len(target_names))), target_names=target_names, zero_division=0))

        cm = confusion_matrix(targets, preds, labels=list(range(len(target_names))))
        cm_normalized = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)

        recalls = np.diag(cm) / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls)

        print("\n" + "=" * 50)
        print("Class-wise Recall")
        print("=" * 50)

        for label, recall in zip(target_names, recalls):
            print(f"{label}: {recall * 100:.2f}%")

        macro_recall = np.mean(recalls) * 100
        print(f"\nMacro Recall: {macro_recall:.2f}%")
    
        save_dir = "./result"
        os.makedirs(save_dir, exist_ok=True)
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        im = ax.imshow(cm)
    
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(target_names)))
    
        ax.set_xticklabels(target_names, rotation=45, ha="right")
        ax.set_yticklabels(target_names)
    
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix (Accuracy: {accuracy:.2f}%)")
    
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
    
        fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")

        plt.close(fig)
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        im = ax.imshow(cm_normalized)
    
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(target_names)))
    
        ax.set_xticklabels(target_names, rotation=45, ha="right")
        ax.set_yticklabels(target_names)
    
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Normalized Confusion Matrix")
    
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                ax.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center")
    
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
    
        fig.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches="tight")
    
        plt.close(fig)
    
        print(f"Saved: {save_dir}/confusion_matrix.png")
        print(f"Saved: {save_dir}/confusion_matrix_normalized.png")
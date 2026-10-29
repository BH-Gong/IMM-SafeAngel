import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import os
import torchaudio
import random
import librosa
import scipy

class BEATsDataset(Dataset):
    def __init__(self, flist_dir, mode="train"):
        self.mode = mode
        self.files = []
        self.targets = []

        flist_files = sorted([f for f in os.listdir(flist_dir) if f.endswith(".flist")])

        class_names = [os.path.splitext(f)[0] for f in flist_files]

        self.label_map = {name: idx for idx, name in enumerate(class_names)}

        for txt_file in flist_files:
            class_name = os.path.splitext(txt_file)[0]
            class_idx = self.label_map[class_name]

            txt_path = os.path.join(flist_dir, txt_file)

            with open(txt_path, "r") as f:
                for line in f:
                    path = line.strip()

                    if not path:
                        continue

                    self.files.append(path)
                    self.targets.append(class_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav = self.load_audio(self.files[idx])
        label = self.targets[idx]

        if isinstance(wav, tuple):
            wav = wav[0]
        
        num_samples = wav.shape[0]

        return {
            "waveform": wav.unsqueeze(0),
            "target": label
        }
        
    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(dim=0)
        return wav
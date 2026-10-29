import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import random

class BEATsDataset(Dataset):
    def __init__(self, file_list, label_map, mode="train"):
        with open(file_list, 'r') as f:
            self.files = [line.strip() for line in f]
        self.mode = mode
        self.label_map = label_map

        self.target_samples = 16000 

        self.targets = []
        for path in self.files:
            parts = path.split("/")
            name_ = [parts[-4], parts[-2]]
            name = '_'.join(name_)
            self.targets.append(self.label_map[name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav = self.load_audio(self.files[idx])
        label = self.targets[idx]

        if isinstance(wav, tuple):
            wav = wav[0]
        
        num_samples = wav.shape[0]
        if num_samples > self.target_samples:
            if self.mode == "train":
                start_idx = random.randint(0, num_samples - self.target_samples)
            else:
                start_idx = 0
            wav = wav[start_idx : start_idx + self.target_samples]
        elif num_samples < self.target_samples:
            pad_size = self.target_samples - num_samples
            wav = F.pad(wav, (0, pad_size), mode='constant', value=0)

        if self.mode == "train":
            rms = torch.sqrt(torch.mean(wav ** 2)) + 1e-6
            noise_level = random.uniform(0.01, 0.04)
            noise = torch.randn_like(wav) * rms * noise_level
            wav = wav + noise

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
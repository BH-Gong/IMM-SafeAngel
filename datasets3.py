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
    def __init__(self, flist_dir, sample_rate, chunk_sec, mode="train"):
        self.mode = mode
        self.files = []
        self.targets = []
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_sec)

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

        if self.mode == "valid":
            self.samples = []
        
            for file_idx, path in enumerate(self.files):
                waveform = self.load_audio(path)
                num_samples = waveform.size(0)
                num_chunks = (num_samples + self.chunk_samples - 1) // self.chunk_samples
                num_samples_to_use = min(2, num_chunks)
        
                chunk_indices = random.sample(range(num_chunks), num_samples_to_use)
        
                for chunk_idx in chunk_indices:
                    self.samples.append((file_idx, chunk_idx))
                    
        if self.mode == 'train':
            self.class_counts = torch.bincount(torch.tensor(self.targets), minlength=len(self.label_map)).tolist()

    def __len__(self):
        if self.mode == "valid":
            return len(self.samples)

        return len(self.files)

    def __getitem__(self, idx):
        if self.mode == "train":
            file_idx = idx
            waveform = self.load_audio(self.files[file_idx])

            label = self.targets[file_idx]
            num_samples = waveform.size(0)

            if num_samples >= self.chunk_samples:
                max_start = (num_samples - self.chunk_samples)

                start = random.randint(0, max_start)

                waveform = waveform[start:start + self.chunk_samples]

            else:
                waveform = F.pad(waveform, (0, self.chunk_samples - num_samples))

            return {
                "waveform": waveform,
                "target": torch.tensor(label, dtype=torch.long)
            }

        elif self.mode == "valid":
            file_idx, chunk_idx = self.samples[idx]
            waveform = self.load_audio(self.files[file_idx])

            label = self.targets[file_idx]

            start = chunk_idx * self.chunk_samples
            end = start + self.chunk_samples

            waveform = waveform[start:end]

            if waveform.size(0) < self.chunk_samples:
                waveform = F.pad(waveform, (0, self.chunk_samples - waveform.size(0)))

            return {
                "waveform": waveform,
                "target": torch.tensor(label, dtype=torch.long),
                "file_idx": file_idx,
                "chunk_idx": chunk_idx
            }

        else:
            waveform = self.load_audio(self.files[idx])
            
            if waveform.size(0) >= self.chunk_samples:
                waveform = waveform[:self.chunk_samples]
                
            else:
                waveform = F.pad(waveform, (0, self.chunk_samples - waveform.size(0)))

            return {
                "waveform": waveform,
                "target": torch.tensor(self.targets[idx], dtype=torch.long)
            }

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        return waveform
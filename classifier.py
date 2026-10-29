import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from BEATs import BEATs, BEATsConfig

class BEATsClassifier(nn.Module):
    def __init__(self, beats_model, num_classes):
        super().__init__()
        self.beats = beats_model
        self.classifier = nn.Linear(768, num_classes)

        for p in self.beats.parameters():
            p.requires_grad = False
        
        for layer in self.beats.encoder.layers[-4:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, wav):
        if wav.dim() == 3:
            wav = wav.squeeze(1)
    
        feats = self.beats.extract_features(wav)[0]
        feats = feats.mean(dim=1)
    
        logits = self.classifier(feats)
    
        return logits

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi


class CNNBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(256, num_classes)

    def preprocess(self, waveform):
        fbanks = []

        for audio in waveform:
            audio = audio.unsqueeze(0) * (2 ** 15)

            fbank = ta_kaldi.fbank(audio, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)

            fbanks.append(fbank)

        fbank = torch.stack(fbanks, dim=0)

        fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)

        return fbank

    def forward(self, waveform):
        fbank = self.preprocess(waveform)

        x = fbank.unsqueeze(1)

        features = self.encoder(x)

        logits = self.classifier(features)

        return logits

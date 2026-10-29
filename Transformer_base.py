import math
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi


class TransformerBaseline(nn.Module):
    def __init__(self, num_classes, n_mels=128, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1,):
        super().__init__()

        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582

        self.input_projection = nn.Linear(n_mels, d_model)

        self.register_buffer("pos_encoding", torch.empty(0))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, 
                                                   activation="gelu", batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def preprocess(self, waveform):
        device = waveform.device
        fbanks = []

        for audio in waveform.detach().cpu():
            audio = audio.unsqueeze(0) * (2 ** 15)

            fbank = ta_kaldi.fbank(audio, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)

        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)

        return fbank.to(device)

    def positional_encoding(self, seq_len, d_model, device, dtype):
        position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        return pe.unsqueeze(0).to(dtype=dtype)

    def forward(self, waveform):
        fbank = self.preprocess(waveform)

        x = self.input_projection(fbank)

        pe = self.positional_encoding(seq_len=x.size(1), d_model=x.size(2), device=x.device, dtype=x.dtype)
        x = x + pe

        x = self.encoder(x)
        x = self.norm(x)

        x = x.mean(dim=1)

        return self.classifier(x)
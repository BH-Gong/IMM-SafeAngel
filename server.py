import io
import json
import torch
import torch.nn.functional as F
import torchaudio

from fastapi import FastAPI, UploadFile, File

from BEATs import BEATs, BEATsConfig
from classifier import BEATsClassifier

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000
TARGET_SAMPLES = 16000

def load_beats(path, device):
    checkpoint = torch.load(path, map_location=device)

    cfg = BEATsConfig(checkpoint["cfg"])

    model = BEATs(cfg)

    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    return model

def load_classifier(beats_model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    label_map = checkpoint["label_map"]

    model = BEATsClassifier(beats_model=beats_model, num_classes=len(label_map))

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, label_map

with open("config.json") as f:
    config = json.load(f)


beats_model = load_beats(config["model_pt_path"], DEVICE)

model, label_map = load_classifier(beats_model, config["classifier_pt_path"], DEVICE)

idx_to_label = {idx: label for label, idx in label_map.items()}

print("Model loaded")
print("Device:", DEVICE)
print("Labels:", label_map)

def preprocess_audio(wav_bytes):
    wav, sr = torchaudio.load(io.BytesIO(wav_bytes))

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    wav = wav.mean(dim=0)

    num_samples = wav.shape[0]

    if num_samples > TARGET_SAMPLES:
        wav = wav[:TARGET_SAMPLES]

    elif num_samples < TARGET_SAMPLES:
        pad_size = TARGET_SAMPLES - num_samples

        wav = F.pad(wav, (0, pad_size), mode="constant", value=0)

    wav = wav.unsqueeze(0)

    return wav

app = FastAPI()

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": str(DEVICE),
        "labels": label_map
    }

import time

@app.post("/infer")
async def infer(file: UploadFile = File(...)):

    t0 = time.perf_counter()

    wav_bytes = await file.read()

    t1 = time.perf_counter()

    waveform = preprocess_audio(wav_bytes)

    t2 = time.perf_counter()

    waveform = waveform.to(DEVICE)

    t3 = time.perf_counter()

    with torch.inference_mode():

        logits = model(waveform)

        torch.cuda.synchronize()

        t4 = time.perf_counter()

        probabilities = torch.softmax(logits, dim=-1)

        pred_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, pred_idx].item()

    t5 = time.perf_counter()

    pred_label = idx_to_label[pred_idx]

    typ = pred_label.split('_')[1]
    if typ == 'normal':
        pred_label_ = typ
    else:
        pred_label_ = pred_label.split('_')[0]

    print(
        f"read={(t1-t0)*1000:.2f} ms | "
        f"preprocess={(t2-t1)*1000:.2f} ms | "
        f"transfer={(t3-t2)*1000:.2f} ms | "
        f"model={(t4-t3)*1000:.2f} ms | "
        f"post={(t5-t4)*1000:.2f} ms | "
        f"total={(t5-t0)*1000:.2f} ms"
    )

    return {
        # "filename": file.filename,
        # "typ": typ,
        # "label": pred_label,
        "anormaly class": pred_label_,
        # "confidence": round(confidence, 4)
    }
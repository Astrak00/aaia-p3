#!/usr/bin/env python3
"""Evaluate all Round 3 trained models and produce final results."""

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report
import os
import numpy as np
from PIL import Image
from collections import Counter

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

DATA_SPLIT_DIR = "data_split"
CLASS_NAMES = ["0-noDR", "1-mild", "2-moderate", "3-severe", "4-proliferativeDR"]
NUM_CLASSES = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SplitImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for class_name in sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        ):
            class_dir = os.path.join(root_dir, class_name)
            label = int(class_name.split("-")[0])
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, self.labels[idx]


def val_t(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def tta_t(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size + 32, size + 32)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_effb0_simple(dropout=0.3):
    m = models.efficientnet_b0(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
    return m


def build_effb0_wide(dropout=0.3):
    m = models.efficientnet_b0(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    return m


def build_effb1_simple(dropout=0.3):
    m = models.efficientnet_b1(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))
    return m


def build_resnet18_wide(dropout=0.4):
    m = models.resnet18(weights=None)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    return m


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds.extend(torch.argmax(model(imgs), 1).cpu().numpy())
            trues.extend(labels.numpy())
    acc = accuracy_score(trues, preds)
    report = classification_report(
        trues, preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    return acc, report


def ensemble_predict(model_list, loader):
    for m in model_list:
        m.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            prob = sum(F.softmax(m(imgs), dim=1) for m in model_list)
            preds.extend(torch.argmax(prob, 1).cpu().numpy())
            trues.extend(labels.numpy())
    acc = accuracy_score(trues, preds)
    report = classification_report(
        trues, preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    return acc, report


val_ds = SplitImageDataset(os.path.join(DATA_SPLIT_DIR, "val"), val_t())
test_ds = SplitImageDataset(os.path.join(DATA_SPLIT_DIR, "test"), val_t())
val_loader = DataLoader(val_ds, 32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, 32, shuffle=False, num_workers=0)

all_results = []

# ---- r3_03 ----
print("\nr3_03_effb0_lr5e4_sampler")
m = build_effb0_simple(0.3).to(device)
m.load_state_dict(
    torch.load("models/best_r3_03_effb0_lr5e4_sampler.pth", map_location=device)
)
va, _ = evaluate(m, val_loader)
ta, rep = evaluate(m, test_loader)
print(f"  val={va:.4f}  test={ta:.4f}")
print(rep)
all_results.append(
    {
        "name": "r3_03_effb0_lr5e4_sampler",
        "config": "EffB0(simple) + AdamW lr=5e-4 + OneCycle + smoothing=0.1 + WeightedSampler",
        "val_acc": va,
        "test_acc": ta,
        "report": rep,
        "checkpoint": "models/best_r3_03_effb0_lr5e4_sampler.pth",
    }
)

# ---- r3_04 ----
print("\nr3_04_effb1_sampler")
m = build_effb1_simple(0.3).to(device)
m.load_state_dict(
    torch.load("models/best_r3_04_effb1_sampler.pth", map_location=device)
)
va, _ = evaluate(m, val_loader)
ta, rep = evaluate(m, test_loader)
print(f"  val={va:.4f}  test={ta:.4f}")
print(rep)
all_results.append(
    {
        "name": "r3_04_effb1_sampler",
        "config": "EffB1(simple head) + AdamW lr=3e-4 + OneCycle + smoothing=0.1 + WeightedSampler",
        "val_acc": va,
        "test_acc": ta,
        "report": rep,
        "checkpoint": "models/best_r3_04_effb1_sampler.pth",
    }
)

# ---- r3_05 ----
print("\nr3_05_effb0_mixup_sampler")
m = build_effb0_simple(0.3).to(device)
m.load_state_dict(
    torch.load("models/best_r3_05_effb0_mixup_sampler.pth", map_location=device)
)
va, _ = evaluate(m, val_loader)
ta, rep = evaluate(m, test_loader)
print(f"  val={va:.4f}  test={ta:.4f}")
print(rep)
all_results.append(
    {
        "name": "r3_05_effb0_mixup_sampler",
        "config": "EffB0(simple) + Mixup(alpha=0.2) + AdamW lr=3e-4 + OneCycle + smoothing=0.1 + WeightedSampler",
        "val_acc": va,
        "test_acc": ta,
        "report": rep,
        "checkpoint": "models/best_r3_05_effb0_mixup_sampler.pth",
    }
)

# ---- r3_06 ----
print("\nr3_06_effb0_focal_sampler")
m = build_effb0_simple(0.3).to(device)
m.load_state_dict(
    torch.load("models/best_r3_06_effb0_focal_sampler.pth", map_location=device)
)
va, _ = evaluate(m, val_loader)
ta, rep = evaluate(m, test_loader)
print(f"  val={va:.4f}  test={ta:.4f}")
print(rep)
all_results.append(
    {
        "name": "r3_06_effb0_focal_sampler",
        "config": "EffB0(simple) + FocalLoss(gamma=2, smoothing=0.1) + AdamW lr=3e-4 + OneCycle + WeightedSampler",
        "val_acc": va,
        "test_acc": ta,
        "report": rep,
        "checkpoint": "models/best_r3_06_effb0_focal_sampler.pth",
    }
)

# ---- Ensemble: r3_04 + r3_03 + r2_01 ----
print("\nr3_07_ensemble_effb1_effb0_r2best")
m_r2_01 = build_effb0_wide(0.3).to(device)
m_r2_01.load_state_dict(
    torch.load(
        "models/best_r2_01_effb0_smoothing_weighted_sampler.pth", map_location=device
    )
)
m_r3_04 = build_effb1_simple(0.3).to(device)
m_r3_04.load_state_dict(
    torch.load("models/best_r3_04_effb1_sampler.pth", map_location=device)
)
m_r3_03 = build_effb0_simple(0.3).to(device)
m_r3_03.load_state_dict(
    torch.load("models/best_r3_03_effb0_lr5e4_sampler.pth", map_location=device)
)

va_e, _ = ensemble_predict([m_r2_01, m_r3_04, m_r3_03], val_loader)
ta_e, rep_e = ensemble_predict([m_r2_01, m_r3_04, m_r3_03], test_loader)
print(f"  val={va_e:.4f}  test={ta_e:.4f}")
print(rep_e)
all_results.append(
    {
        "name": "r3_07_ensemble_r2_01_r3_04_r3_03",
        "config": "Ensemble(r2_01_effb0_wide + r3_04_effb1 + r3_03_effb0) avg softmax",
        "val_acc": va_e,
        "test_acc": ta_e,
        "report": rep_e,
        "checkpoint": "ensemble",
    }
)

# ---- Ensemble: r2_01 + exp5 + exp2 (from r3_02) ----
m_exp5 = build_effb0_simple(0.3).to(device)
m_exp5.load_state_dict(
    torch.load("models/best_exp5_efficientnetb0_smoothing.pth", map_location=device)
)
m_exp2 = build_resnet18_wide(0.4).to(device)
m_exp2.load_state_dict(
    torch.load("models/best_exp2_resnet18_label_smoothing.pth", map_location=device)
)

print("\nr3_02_ensemble_r2_01_exp5_exp2")
va_e2, _ = ensemble_predict([m_r2_01, m_exp5, m_exp2], val_loader)
ta_e2, rep_e2 = ensemble_predict([m_r2_01, m_exp5, m_exp2], test_loader)
print(f"  val={va_e2:.4f}  test={ta_e2:.4f}")
print(rep_e2)
all_results.append(
    {
        "name": "r3_02_ensemble_r2_01_exp5_exp2",
        "config": "Ensemble(r2_01_effb0_wide + exp5_effb0 + exp2_resnet18) avg softmax",
        "val_acc": va_e2,
        "test_acc": ta_e2,
        "report": rep_e2,
        "checkpoint": "ensemble",
    }
)

# ---- Best ensemble ever: 4 models ----
print("\nr3_08_ensemble_4models")
m_r2_06 = build_effb0_wide(0.3).to(device)
m_r2_06.load_state_dict(
    torch.load("models/best_r2_06_effb0_sgd_nesterov.pth", map_location=device)
)

va_e4, _ = ensemble_predict([m_r2_01, m_exp5, m_exp2, m_r2_06], val_loader)
ta_e4, rep_e4 = ensemble_predict([m_r2_01, m_exp5, m_exp2, m_r2_06], test_loader)
print(f"  val={va_e4:.4f}  test={ta_e4:.4f}")
print(rep_e4)
all_results.append(
    {
        "name": "r3_08_ensemble_4models",
        "config": "Ensemble(r2_01 + exp5 + exp2 + r2_06) 4-model avg softmax",
        "val_acc": va_e4,
        "test_acc": ta_e4,
        "report": rep_e4,
        "checkpoint": "ensemble",
    }
)

# Final summary
all_results.sort(key=lambda x: x["test_acc"], reverse=True)
print(f"\n{'=' * 70}")
print("ROUND 3 RANKINGS BY TEST ACCURACY")
print(f"{'=' * 70}")
print(f"{'Rank':>4} {'Name':>52} {'Val Acc':>9} {'Test Acc':>9}")
print("-" * 78)
for i, r in enumerate(all_results):
    print(f"{i + 1:>4} {r['name']:>52} {r['val_acc']:>9.4f} {r['test_acc']:>9.4f}")

print(f"\nBest R3: {all_results[0]['name']}  test_acc={all_results[0]['test_acc']:.4f}")
print("Prev best (R2): r2_01_effb0_smoothing_weighted_sampler  test_acc=0.5276")

import json

with open("r3_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved to r3_results.json")

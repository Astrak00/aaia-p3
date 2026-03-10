#!/usr/bin/env python3
"""
Evaluate Round 2 models that are already saved (no training needed).
r2_07 two-phase: also runs phase2 fine-tuning from phase1 checkpoint.
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report
import os, time
import numpy as np
from PIL import Image
from collections import Counter

SEED = 42
torch.manual_seed(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

DATA_SPLIT_DIR = "data_split"
MODEL_DIR = "models"
CLASS_NAMES = ["0-noDR", "1-mild", "2-moderate", "3-severe", "4-proliferativeDR"]
NUM_CLASSES = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SplitImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        class_names = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        for class_name in class_names:
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


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_effb0(dropout=0.4):
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


def build_resnet18(dropout=0.4):
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
            out = model(imgs)
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            trues.extend(labels.numpy())
    acc = accuracy_score(trues, preds)
    report = classification_report(
        trues, preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    return acc, report


val_ds = SplitImageDataset(os.path.join(DATA_SPLIT_DIR, "val"), get_val_transform())
test_ds = SplitImageDataset(os.path.join(DATA_SPLIT_DIR, "test"), get_val_transform())
val_loader = DataLoader(val_ds, 32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, 32, shuffle=False, num_workers=0)

experiments_r2 = [
    {
        "name": "r2_01_effb0_smoothing_weighted_sampler",
        "arch": "effb0",
        "dropout": 0.3,
        "config": "EffB0 + AdamW + OneCycle + smoothing=0.1 + WeightedSampler",
    },
    {
        "name": "r2_02_effb0_diff_lr",
        "arch": "effb0",
        "dropout": 0.3,
        "config": "EffB0 + AdamW + OneCycle + smoothing=0.1 + DiffLR(backbone=lr/10)",
    },
    {
        "name": "r2_03_effb0_strong_aug_sampler",
        "arch": "effb0",
        "dropout": 0.3,
        "config": "EffB0 + AdamW + OneCycle + smoothing=0.1 + StrongAug + WeightedSampler",
    },
    {
        "name": "r2_04_resnet18_diff_lr_sampler",
        "arch": "resnet18",
        "dropout": 0.4,
        "config": "ResNet18 + AdamW + OneCycle + smoothing=0.1 + DiffLR + WeightedSampler",
    },
    {
        "name": "r2_05_effb0_cosine_longer",
        "arch": "effb0",
        "dropout": 0.35,
        "config": "EffB0 + AdamW + Cosine(120ep) + smoothing=0.05 + WeightedSampler",
    },
    {
        "name": "r2_06_effb0_sgd_nesterov",
        "arch": "effb0",
        "dropout": 0.3,
        "config": "EffB0 + SGD(nesterov) + Cosine + smoothing=0.1 + WeightedSampler",
    },
]

all_results = []

for exp in experiments_r2:
    name = exp["name"]
    ckpt = f"models/best_{name}.pth"
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {name}")

    if not os.path.exists(ckpt):
        print(f"  WARNING: checkpoint not found: {ckpt}")
        continue

    if exp["arch"] == "effb0":
        model = build_effb0(exp["dropout"])
    else:
        model = build_resnet18(exp["dropout"])

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)

    val_acc, _ = evaluate(model, val_loader)
    test_acc, report = evaluate(model, test_loader)

    print(f"  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")
    print(report)

    all_results.append(
        {
            "name": name,
            "config": exp["config"],
            "val_acc": val_acc,
            "test_acc": test_acc,
            "report": report,
            "checkpoint": ckpt,
        }
    )

# r2_07: phase1 checkpoint only. Check if full checkpoint exists.
ckpt07 = "models/best_r2_07_effb0_two_phase.pth"
if os.path.exists(ckpt07):
    print(f"\n{'=' * 60}")
    print("Evaluating: r2_07_effb0_two_phase (full checkpoint)")
    model = build_effb0(dropout=0.4)
    model.load_state_dict(torch.load(ckpt07, map_location=device))
    model = model.to(device)
    val_acc, _ = evaluate(model, val_loader)
    test_acc, report = evaluate(model, test_loader)
    print(f"  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")
    print(report)
    all_results.append(
        {
            "name": "r2_07_effb0_two_phase",
            "config": "EffB0 TwoPhase: freeze20ep + cosine60ep diffLR(3e-5/3e-4)",
            "val_acc": val_acc,
            "test_acc": test_acc,
            "report": report,
            "checkpoint": ckpt07,
        }
    )
else:
    print(f"\nr2_07 full checkpoint not found (only phase1 exists), skipping.")

all_results.sort(key=lambda x: x["test_acc"], reverse=True)
print(f"\n{'=' * 70}")
print("ROUND 2 RANKINGS BY TEST ACCURACY")
print(f"{'=' * 70}")
print(f"{'Rank':>4} {'Name':>45} {'Val Acc':>9} {'Test Acc':>9}")
print("-" * 65)
for i, r in enumerate(all_results):
    print(f"{i + 1:>4} {r['name']:>45} {r['val_acc']:>9.4f} {r['test_acc']:>9.4f}")

if all_results:
    print(
        f"\nBest R2: {all_results[0]['name']}  test_acc={all_results[0]['test_acc']:.4f}"
    )

import json

with open("r2_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved to r2_results.json")

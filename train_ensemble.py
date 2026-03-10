"""
Train two companion models for the ensemble:
  1. exp5-style: EffB0 simple head + AdamW + label smoothing (no weighted sampler)
  2. exp2-style: ResNet18 wide head + AdamW + label smoothing + WeightedSampler
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

from sklearn.metrics import accuracy_score

import random, os, time
import numpy as np
from PIL import Image
from collections import Counter

os.chdir(
    "/Users/edu/Documents/Universidad/Master_Ingeniería_Informática/clases/iaiao/p3"
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

CLASS_NAMES = ["0-noDR", "1-mild", "2-moderate", "3-severe", "4-proliferativeDR"]
NUM_CLASSES = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
data_split_dir = "data_split"
model_dir = "models"


class SplitImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths, self.labels = [], []
        for cls in sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        ):
            for f in sorted(os.listdir(os.path.join(root_dir, cls))):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root_dir, cls, f))
                    self.labels.append(int(cls.split("-")[0]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, self.labels[idx]


train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

train_dataset = SplitImageDataset(
    os.path.join(data_split_dir, "train"), train_transform
)
val_dataset = SplitImageDataset(os.path.join(data_split_dir, "val"), val_transform)
test_dataset = SplitImageDataset(os.path.join(data_split_dir, "test"), val_transform)

label_counts = Counter(train_dataset.labels)
total = len(train_dataset.labels)
class_weights = torch.zeros(NUM_CLASSES)
for i in range(NUM_CLASSES):
    class_weights[i] = total / (NUM_CLASSES * label_counts.get(i, 1))
class_weights_dev = class_weights.to(device)

sample_weights = [1.0 / label_counts[lbl] for lbl in train_dataset.labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

BATCH_SIZE = 32


def evaluate(model, loader, criterion, device):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(true_labels, preds)


def train_model(
    name, model, use_sampler=True, num_epochs=35, patience=10, lr=3e-4, wd=1e-4
):
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print(f"{'=' * 60}")
    model = model.to(device)

    if use_sampler:
        loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
        )
    else:
        loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_dev, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(loader),
        epochs=num_epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    save_path = os.path.join(model_dir, f"best_{name}.pth")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"  Epoch  TrainLoss   ValLoss   ValAcc")
    print("-" * 44)

    t0 = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            marker = " *" if improved else ""
            print(
                f"  {epoch + 1:>4}/{num_epochs}  {avg_loss:>9.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}{marker}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    elapsed = time.time() - t0
    print(f"Done in {elapsed / 60:.1f} min. Best val acc: {best_val_acc:.4f}")
    return save_path, best_val_acc


# --- Model 1: EffB0 simple head, no weighted sampler (diverse from main model) ---
def build_effb0_simple():
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, NUM_CLASSES))
    return m


# --- Model 2: ResNet18 wide head + weighted sampler ---
def build_resnet18_wide():
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    return m


path1, val1 = train_model("exp5_effb0_simple", build_effb0_simple(), use_sampler=False)
path2, val2 = train_model("exp2_resnet18_wide", build_resnet18_wide(), use_sampler=True)

print(f"\nModel 1 (EffB0 simple): val={val1:.4f}")
print(f"Model 2 (ResNet18 wide): val={val2:.4f}")

# Quick ensemble test
print("\n--- Ensemble test ---")
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

main_model_path = os.path.join(model_dir, "best_model_effb0_sampler.pth")


def build_main_effb0():
    m = models.efficientnet_b0(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
    return m


m0 = build_main_effb0().to(device)
m0.load_state_dict(torch.load(main_model_path, map_location=device))

m1 = build_effb0_simple().to(device)
m1.load_state_dict(torch.load(path1, map_location=device))

m2 = build_resnet18_wide().to(device)
m2.load_state_dict(torch.load(path2, map_location=device))

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

criterion_test = nn.CrossEntropyLoss(weight=class_weights_dev, label_smoothing=0.1)

for m in [m0, m1, m2]:
    m.eval()

ens_preds, ens_true = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        prob_sum = sum(F.softmax(m(images), dim=1) for m in [m0, m1, m2])
        ens_preds.extend(torch.argmax(prob_sum, 1).cpu().numpy())
        ens_true.extend(labels.numpy())

ens_acc = accuracy_score(ens_true, ens_preds)
print(f"Ensemble (3 models) Test Accuracy: {ens_acc:.4f}")

import json

json.dump(
    {"ens_acc": ens_acc, "val1": val1, "val2": val2}, open("ensemble_results.json", "w")
)
print("Saved ensemble_results.json")

"""
train_round6.py

Round 6 experiments: pushing accuracy beyond 0.6309 on data_preprocessed/.

Experiments:
  r6_01 — EffB0, 80 epochs (model was still improving at ep40, never early-stopped)
  r6_02 — EfficientNetB2 (slightly larger backbone, 9M params, 1408 features)
  r6_03 — EffB0 + Mixup augmentation (alpha=0.4, great for ordinal DR grading)
  r6_04 — EffB0 + larger 320px crop from 512px images (more detail)

All configs share:
  - data_preprocessed/ (512x512 preprocessed fundus images)
  - AdamW lr=3e-4, weight_decay=1e-4
  - OneCycleLR (pct_start=0.1, cosine)
  - CrossEntropyLoss(label_smoothing=0.1) + inverse-freq class weights
  - No WeightedRandomSampler
  - Gradient clipping max_norm=1.0
  - SEED=42
"""

import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
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
print(f"Device: {device}")

DATA_DIR = "data_preprocessed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ["0-noDR", "1-mild", "2-moderate", "3-severe", "4-proliferativeDR"]
NUM_CLASSES = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class SplitImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths, self.labels = [], []
        for cls in sorted(os.listdir(root_dir)):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = int(cls.split("-")[0])
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)
        print(f"  Loaded {len(self.image_paths)} images from {root_dir}")
        c = Counter(self.labels)
        for name in sorted(os.listdir(root_dir)):
            if os.path.isdir(os.path.join(root_dir, name)):
                idx = int(name.split("-")[0])
                print(f"    {name}: {c[idx]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, self.labels[idx]


def make_transforms(crop_size=224):
    """Standard augmentation pipeline for preprocessed 512×512 images."""
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.4, saturation=0.3, hue=0.05
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


# ─────────────────────────────────────────────────────────────────────────────
# Mixup
# ─────────────────────────────────────────────────────────────────────────────


def mixup_data(x, y, alpha=0.4):
    """Apply Mixup augmentation to a batch. Returns mixed_x, y_a, y_b, lam."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


def build_effb0(dropout=0.3):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
    )
    return m


def build_effb2(dropout=0.3):
    m = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Training / Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            total_loss += criterion(out, labels).item()
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(trues, preds), trues, preds


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    num_epochs,
    patience,
    save_path,
    use_mixup=False,
    mixup_alpha=0.4,
    lr=3e-4,
    weight_decay=1e-4,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_acc = 0.0
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(
        f"  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>8}  {'ValAcc':>7}  {'Best':>7}"
    )
    print("  " + "-" * 50)
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_mixup:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha)
                loss = mixup_criterion(criterion, model(imgs), y_a, y_b, lam)
            else:
                loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            marker = " *" if improved else ""
            print(
                f"  {epoch + 1:>4}/{num_epochs}  {avg_loss:>10.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}  {best_val_acc:>7.4f}{marker}"
            )

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch + 1}.")
            break

    elapsed = time.time() - t0
    print(
        f"\n  Training finished in {elapsed / 60:.1f} min  |  Best val acc: {best_val_acc:.4f}"
    )
    return history, best_val_acc, elapsed / 60


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(cfg, train_ds, val_ds, test_ds):
    name = cfg["name"]
    num_epochs = cfg["num_epochs"]
    patience = cfg["patience"]
    batch_size = cfg["batch_size"]
    use_mixup = cfg.get("use_mixup", False)
    mixup_alpha = cfg.get("mixup_alpha", 0.4)
    lr = cfg.get("lr", 3e-4)
    save_path = os.path.join(MODEL_DIR, f"{name}.pth")

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {name}")
    print(f"  {cfg.get('desc', '')}")
    print(f"{'=' * 70}")

    # Build model
    model = cfg["build_fn"]().to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    # Data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Class weights
    lc = Counter(train_ds.labels)
    total = len(train_ds.labels)
    cw = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        cw[i] = total / (NUM_CLASSES * lc.get(i, 1))
    criterion = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=0.1)

    # Train
    history, best_val_acc, elapsed_min = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        num_epochs=num_epochs,
        patience=patience,
        save_path=save_path,
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        lr=lr,
    )

    # Test
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, test_acc, trues, preds = evaluate(model, test_loader, criterion)
    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(classification_report(trues, preds, target_names=CLASS_NAMES, digits=4))
    print("  Confusion Matrix:")
    print(confusion_matrix(trues, preds))

    return {
        "name": name,
        "desc": cfg.get("desc", ""),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "elapsed_min": elapsed_min,
        "history": history,
        "classification_report": classification_report(
            trues, preds, target_names=CLASS_NAMES, digits=4
        ),
        "confusion_matrix": confusion_matrix(trues, preds).tolist(),
    }


def main():
    # ── Load datasets ──────────────────────────────────────────────────────
    print("\nLoading datasets from data_preprocessed/...")

    # r6_01, r6_02, r6_03 use 224px crop
    train_tf_224, val_tf_224 = make_transforms(crop_size=224)
    # r6_04 uses 320px crop
    train_tf_320, val_tf_320 = make_transforms(crop_size=320)

    print("Train (224):")
    train_ds_224 = SplitImageDataset(os.path.join(DATA_DIR, "train"), train_tf_224)
    print("Val (224):")
    val_ds_224 = SplitImageDataset(os.path.join(DATA_DIR, "val"), val_tf_224)
    print("Test (224):")
    test_ds_224 = SplitImageDataset(os.path.join(DATA_DIR, "test"), val_tf_224)

    print("Train (320):")
    train_ds_320 = SplitImageDataset(os.path.join(DATA_DIR, "train"), train_tf_320)
    print("Val (320):")
    val_ds_320 = SplitImageDataset(os.path.join(DATA_DIR, "val"), val_tf_320)
    print("Test (320):")
    test_ds_320 = SplitImageDataset(os.path.join(DATA_DIR, "test"), val_tf_320)

    # ── Experiment configs ─────────────────────────────────────────────────
    experiments = [
        {
            "name": "r6_01_effb0_80ep",
            "desc": "EffB0, 80 epochs / patience=20. Model was still improving at ep40.",
            "build_fn": lambda: build_effb0(dropout=0.3),
            "num_epochs": 80,
            "patience": 20,
            "batch_size": 32,
            "train_ds": train_ds_224,
            "val_ds": val_ds_224,
            "test_ds": test_ds_224,
        },
        {
            "name": "r6_02_effb2_224",
            "desc": "EfficientNetB2 (9M params, 1408 features). Slightly larger than B0.",
            "build_fn": lambda: build_effb2(dropout=0.3),
            "num_epochs": 60,
            "patience": 15,
            "batch_size": 32,
            "train_ds": train_ds_224,
            "val_ds": val_ds_224,
            "test_ds": test_ds_224,
        },
        {
            "name": "r6_03_effb0_mixup",
            "desc": "EffB0 + Mixup (alpha=0.4). Mixup helps ordinal DR grading.",
            "build_fn": lambda: build_effb0(dropout=0.3),
            "num_epochs": 80,
            "patience": 20,
            "batch_size": 32,
            "use_mixup": True,
            "mixup_alpha": 0.4,
            "train_ds": train_ds_224,
            "val_ds": val_ds_224,
            "test_ds": test_ds_224,
        },
        {
            "name": "r6_04_effb0_320px",
            "desc": "EffB0 + 320px crop (vs 224px). More detail from 512px preprocessed images.",
            "build_fn": lambda: build_effb0(dropout=0.3),
            "num_epochs": 60,
            "patience": 15,
            "batch_size": 16,  # smaller batch to fit 320px images in MPS memory
            "train_ds": train_ds_320,
            "val_ds": val_ds_320,
            "test_ds": test_ds_320,
        },
    ]

    results = []
    for cfg in experiments:
        # pull dataset refs out of cfg for run_experiment signature
        train_ds = cfg.pop("train_ds")
        val_ds = cfg.pop("val_ds")
        test_ds = cfg.pop("test_ds")
        res = run_experiment(cfg, train_ds, val_ds, test_ds)
        results.append(res)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("ROUND 6 SUMMARY")
    print("=" * 70)
    print(f"  {'Name':<35}  {'ValAcc':>7}  {'TestAcc':>8}  {'Time(m)':>8}")
    print("  " + "-" * 65)
    for r in sorted(results, key=lambda x: x["test_acc"], reverse=True):
        print(
            f"  {r['name']:<35}  {r['best_val_acc']:>7.4f}  {r['test_acc']:>8.4f}  {r['elapsed_min']:>8.1f}"
        )

    # Save JSON
    with open("train_round6_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to train_round6_results.json")


if __name__ == "__main__":
    main()

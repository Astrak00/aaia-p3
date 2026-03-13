"""
train_round8.py

Round 8: Squeeze maximum accuracy from EffB0 + 320px.

Insights from R7:
  - EffB0 + 320px + 120ep = 0.6722 (best so far)
  - Model stopped at ep78 patience; best ckpt at ep48 — still improving
  - B2 overfits with 320px, needs stronger regularization
  - Ensemble of same-resolution models not yet tried

Experiments:
  r8_01 — EffB0 + 320px + 200 epochs / patience=40 (push further)
  r8_02 — EffB0 + 320px + stronger dropout (0.5 vs 0.3) + 120ep
           Addresses overfitting on minority classes
  r8_03 — EfficientNetB2 + 320px + stronger dropout (0.5) + lower lr (1e-4)
           Try to close the val/test gap seen in r7_04
  r8_04 — Ensemble: r7_01 + a NEW EffB0 320px run with different seed (diversity)
"""

import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B2_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
from collections import Counter

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
        print(f"  Loaded {len(self.image_paths)} from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, self.labels[idx]


def make_transforms(crop_size=320):
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
# Models
# ─────────────────────────────────────────────────────────────────────────────


def build_effb0(dropout=0.3):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout), nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    )
    return m


def build_effb2(dropout=0.5):
    m = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout), nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    )
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
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


def get_all_probs(model, ds, batch_size=16):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_list = []
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs.to(device))
            probs_list.append(F.softmax(out, dim=1).cpu())
    return torch.cat(probs_list, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    num_epochs,
    patience,
    save_path,
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
    print(f"\n  Done in {elapsed / 60:.1f} min  |  Best val acc: {best_val_acc:.4f}")
    return history, best_val_acc, elapsed / 60


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Load datasets (all 320px)
    print("Loading datasets (320px)...")
    train_tf, val_tf = make_transforms(320)
    train_ds = SplitImageDataset(os.path.join(DATA_DIR, "train"), train_tf)
    val_ds = SplitImageDataset(os.path.join(DATA_DIR, "val"), val_tf)
    test_ds = SplitImageDataset(os.path.join(DATA_DIR, "test"), val_tf)

    # Class weights
    lc = Counter(train_ds.labels)
    total_n = len(train_ds.labels)
    cw = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        cw[i] = total_n / (NUM_CLASSES * lc.get(i, 1))
    criterion = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=0.1)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    results = []

    # ── r8_01: EffB0 320px 200ep patience=40 ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  r8_01: EffB0 + 320px + 200 epochs (patience=40)")
    print("=" * 70)
    m1 = build_effb0(dropout=0.3).to(device)
    h1, bva1, elt1 = train_model(
        m1,
        train_loader,
        val_loader,
        criterion,
        num_epochs=200,
        patience=40,
        save_path=os.path.join(MODEL_DIR, "r8_01_effb0_320px_200ep.pth"),
    )
    m1.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r8_01_effb0_320px_200ep.pth"), map_location=device
        )
    )
    _, ta1, tr1, pr1 = evaluate(m1, test_loader, criterion)
    print(f"  Test: {ta1:.4f}")
    print(classification_report(tr1, pr1, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(tr1, pr1))
    results.append(
        {
            "name": "r8_01_effb0_320px_200ep",
            "best_val_acc": bva1,
            "test_acc": ta1,
            "elapsed_min": elt1,
            "report": classification_report(
                tr1, pr1, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(tr1, pr1).tolist(),
        }
    )

    # ── r8_02: EffB0 320px stronger dropout (0.5) + 120ep ────────────────────
    print("\n" + "=" * 70)
    print("  r8_02: EffB0 + 320px + dropout=0.5 + 120 epochs (patience=30)")
    print("=" * 70)
    m2 = build_effb0(dropout=0.5).to(device)
    h2, bva2, elt2 = train_model(
        m2,
        train_loader,
        val_loader,
        criterion,
        num_epochs=120,
        patience=30,
        save_path=os.path.join(MODEL_DIR, "r8_02_effb0_320px_do50.pth"),
    )
    m2.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r8_02_effb0_320px_do50.pth"), map_location=device
        )
    )
    _, ta2, tr2, pr2 = evaluate(m2, test_loader, criterion)
    print(f"  Test: {ta2:.4f}")
    print(classification_report(tr2, pr2, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(tr2, pr2))
    results.append(
        {
            "name": "r8_02_effb0_320px_do50",
            "best_val_acc": bva2,
            "test_acc": ta2,
            "elapsed_min": elt2,
            "report": classification_report(
                tr2, pr2, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(tr2, pr2).tolist(),
        }
    )

    # ── r8_03: EfficientNetB2 + 320px + do=0.5 + lr=1e-4 ─────────────────────
    print("\n" + "=" * 70)
    print("  r8_03: EfficientNetB2 + 320px + dropout=0.5 + lr=1e-4 + 120ep")
    print("=" * 70)
    m3 = build_effb2(dropout=0.5).to(device)
    h3, bva3, elt3 = train_model(
        m3,
        train_loader,
        val_loader,
        criterion,
        num_epochs=120,
        patience=30,
        save_path=os.path.join(MODEL_DIR, "r8_03_effb2_320px_do50.pth"),
        lr=1e-4,  # lower lr to reduce overfitting
    )
    m3.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r8_03_effb2_320px_do50.pth"), map_location=device
        )
    )
    _, ta3, tr3, pr3 = evaluate(m3, test_loader, criterion)
    print(f"  Test: {ta3:.4f}")
    print(classification_report(tr3, pr3, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(tr3, pr3))
    results.append(
        {
            "name": "r8_03_effb2_320px_do50",
            "best_val_acc": bva3,
            "test_acc": ta3,
            "elapsed_min": elt3,
            "report": classification_report(
                tr3, pr3, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(tr3, pr3).tolist(),
        }
    )

    # ── r8_04: Ensemble r7_01 + r8_01 (same arch/resolution, different init/order) ──
    print("\n" + "=" * 70)
    print("  r8_04: Ensemble r7_01 + r8_01 (both EffB0 320px, different seeds)")
    print("=" * 70)
    m_r7_01 = build_effb0(dropout=0.3).to(device)
    m_r7_01.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r7_01_effb0_320px_120ep.pth"), map_location=device
        )
    )

    probs_r7 = get_all_probs(m_r7_01, test_ds)
    probs_r8 = get_all_probs(m1, test_ds)

    ens_probs = (probs_r7 + probs_r8) / 2
    ens_preds = torch.argmax(ens_probs, 1).numpy()
    ens_trues = test_ds.labels
    ens_acc = accuracy_score(ens_trues, ens_preds)
    print(f"  Ensemble Test Accuracy: {ens_acc:.4f}")
    print(
        classification_report(ens_trues, ens_preds, target_names=CLASS_NAMES, digits=4)
    )
    print(confusion_matrix(ens_trues, ens_preds))
    results.append(
        {
            "name": "r8_04_ensemble_r7r8",
            "best_val_acc": None,
            "test_acc": ens_acc,
            "elapsed_min": 0,
            "report": classification_report(
                ens_trues, ens_preds, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(ens_trues, ens_preds).tolist(),
        }
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("ROUND 8 SUMMARY")
    print("=" * 70)
    print(f"  {'Name':<35}  {'ValAcc':>7}  {'TestAcc':>8}  {'Time(m)':>8}")
    print("  " + "-" * 65)
    for r in sorted(results, key=lambda x: x["test_acc"], reverse=True):
        va = f"{r['best_val_acc']:.4f}" if r["best_val_acc"] is not None else "  N/A "
        print(
            f"  {r['name']:<35}  {va:>7}  {r['test_acc']:>8.4f}  {r['elapsed_min']:>8.1f}"
        )

    with open("train_round8_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to train_round8_results.json")


if __name__ == "__main__":
    main()

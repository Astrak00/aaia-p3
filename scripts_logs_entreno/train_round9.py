"""
train_round9.py

Round 9: Push accuracy further with TTA, longer training, and stronger dropout.

Insights from R8:
  - r8_01 EffB0+320px+200ep = 0.7010 (best single); stopped at ep~140 on patience=40
  - r7_01 + r8_01 ensemble = 0.7072 (best overall)
  - r8_02 dropout=0.5 only trained 120ep — could still be improving
  - TTA (test-time augmentation) is free — try on best ensemble and best single

Experiments:
  r9_00 — TTA evaluation on r8_01 (single best) and r7_01+r8_01 ensemble (free, no training)
  r9_01 — EffB0 + 320px + 300 epochs / patience=60  (model likely still improving at ep200)
  r9_02 — EffB0 + 320px + dropout=0.5 + 200 epochs  (r8_02 was only 120ep)
  r9_03 — EfficientNetB2 + 320px + dropout=0.5 + 200ep + lr=1e-4  (fix r7/r8 overfitting)
  [Ensemble evals after each training completes]
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

CROP_SIZE = 320


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


def make_transforms(crop_size=CROP_SIZE):
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


def make_tta_transforms(crop_size=CROP_SIZE):
    """
    8-fold TTA: original + 3 rotations + horizontal flip of each.
    Each augmentation is a deterministic transform applied separately.
    Returns a list of transform pipelines.
    """
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    tta_list = []
    for angle in [0, 90, 180, 270]:
        # Original orientation
        tf = transforms.Compose(
            [
                transforms.Resize((crop_size, crop_size)),
                transforms.RandomRotation(degrees=(angle, angle)),
                normalize,
            ]
        )
        tta_list.append(tf)
        # Horizontal flip
        tf_flip = transforms.Compose(
            [
                transforms.Resize((crop_size, crop_size)),
                transforms.RandomRotation(degrees=(angle, angle)),
                transforms.RandomHorizontalFlip(p=1.0),
                normalize,
            ]
        )
        tta_list.append(tf_flip)
    return tta_list  # 8 transforms


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
    """Standard inference — returns softmax probs for every sample."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_list = []
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs.to(device))
            probs_list.append(F.softmax(out, dim=1).cpu())
    return torch.cat(probs_list, dim=0)  # (N, 5)


def get_tta_probs(model, image_paths, labels, tta_transforms, batch_size=16):
    """
    TTA inference: for each TTA transform, run a full forward pass,
    then average all softmax outputs.
    """
    model.eval()
    accumulated = None
    for tf in tta_transforms:
        # Build a fresh dataset with this TTA transform
        class _TFDataset(Dataset):
            def __init__(self, paths, lbls, transform):
                self.paths = paths
                self.labels = lbls
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                with Image.open(self.paths[idx]) as img:
                    img = img.convert("RGB")
                    img = self.transform(img)
                return img, self.labels[idx]

        ds = _TFDataset(image_paths, labels, tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        probs_list = []
        with torch.no_grad():
            for imgs, _ in loader:
                out = model(imgs.to(device))
                probs_list.append(F.softmax(out, dim=1).cpu())
        batch_probs = torch.cat(probs_list, dim=0)
        if accumulated is None:
            accumulated = batch_probs
        else:
            accumulated = accumulated + batch_probs

    return accumulated / len(tta_transforms)


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

        if (epoch + 1) % 10 == 0 or epoch == 0 or improved:
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
    print("Loading datasets...")
    train_tf, val_tf = make_transforms(CROP_SIZE)
    tta_tfs = make_tta_transforms(CROP_SIZE)
    print(f"  TTA: {len(tta_tfs)} transforms (4 rotations × 2 flips)")

    train_ds = SplitImageDataset(os.path.join(DATA_DIR, "train"), train_tf)
    val_ds = SplitImageDataset(os.path.join(DATA_DIR, "val"), val_tf)
    test_ds = SplitImageDataset(os.path.join(DATA_DIR, "test"), val_tf)

    # Class weights (inverse-frequency)
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

    # ── r9_00: TTA on existing best models (no training) ─────────────────────
    print("\n" + "=" * 70)
    print("  r9_00: TTA evaluation on r8_01 (single) and r7+r8 ensemble")
    print("=" * 70)

    m_r8_01 = build_effb0(dropout=0.3).to(device)
    m_r8_01.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r8_01_effb0_320px_200ep.pth"), map_location=device
        )
    )
    m_r7_01 = build_effb0(dropout=0.3).to(device)
    m_r7_01.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r7_01_effb0_320px_120ep.pth"), map_location=device
        )
    )

    test_paths = test_ds.image_paths
    test_labels = test_ds.labels

    # TTA on r8_01 single
    print("\n  --- TTA: r8_01 single model ---")
    t_tta = time.time()
    tta_probs_r8 = get_tta_probs(m_r8_01, test_paths, test_labels, tta_tfs)
    tta_preds_r8 = torch.argmax(tta_probs_r8, 1).numpy()
    tta_acc_r8 = accuracy_score(test_labels, tta_preds_r8)
    print(f"  r8_01 standard:  0.7010")
    print(f"  r8_01 TTA:       {tta_acc_r8:.4f}  ({time.time() - t_tta:.1f}s)")
    print(
        classification_report(
            test_labels, tta_preds_r8, target_names=CLASS_NAMES, digits=4
        )
    )
    print(confusion_matrix(test_labels, tta_preds_r8))

    results.append(
        {
            "name": "r9_00a_tta_r8_01",
            "best_val_acc": None,
            "test_acc": tta_acc_r8,
            "elapsed_min": (time.time() - t_tta) / 60,
            "report": classification_report(
                test_labels, tta_preds_r8, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(test_labels, tta_preds_r8).tolist(),
        }
    )

    # TTA on ensemble r7_01 + r8_01
    print("\n  --- TTA: ensemble r7_01 + r8_01 ---")
    t_tta2 = time.time()
    tta_probs_r7 = get_tta_probs(m_r7_01, test_paths, test_labels, tta_tfs)
    tta_ens_probs = (tta_probs_r7 + tta_probs_r8) / 2
    tta_ens_preds = torch.argmax(tta_ens_probs, 1).numpy()
    tta_ens_acc = accuracy_score(test_labels, tta_ens_preds)
    print(f"  Ensemble standard:  0.7072")
    print(
        f"  Ensemble TTA:       {tta_ens_acc:.4f}  ({time.time() - t_tta2:.1f}s for r7)"
    )
    print(
        classification_report(
            test_labels, tta_ens_preds, target_names=CLASS_NAMES, digits=4
        )
    )
    print(confusion_matrix(test_labels, tta_ens_preds))

    results.append(
        {
            "name": "r9_00b_tta_ensemble_r7r8",
            "best_val_acc": None,
            "test_acc": tta_ens_acc,
            "elapsed_min": (time.time() - t_tta2) / 60,
            "report": classification_report(
                test_labels, tta_ens_preds, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(test_labels, tta_ens_preds).tolist(),
        }
    )

    # Also try TTA + r8_02 in the ensemble
    print("\n  --- TTA: ensemble r7_01 + r8_01 + r8_02 ---")
    m_r8_02 = build_effb0(dropout=0.5).to(device)
    m_r8_02.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r8_02_effb0_320px_do50.pth"), map_location=device
        )
    )
    tta_probs_r8_02 = get_tta_probs(m_r8_02, test_paths, test_labels, tta_tfs)
    tta_3ens_probs = (tta_probs_r7 + tta_probs_r8 + tta_probs_r8_02) / 3
    tta_3ens_preds = torch.argmax(tta_3ens_probs, 1).numpy()
    tta_3ens_acc = accuracy_score(test_labels, tta_3ens_preds)
    print(f"  3-model TTA ensemble: {tta_3ens_acc:.4f}")
    print(
        classification_report(
            test_labels, tta_3ens_preds, target_names=CLASS_NAMES, digits=4
        )
    )
    print(confusion_matrix(test_labels, tta_3ens_preds))

    results.append(
        {
            "name": "r9_00c_tta_3ensemble",
            "best_val_acc": None,
            "test_acc": tta_3ens_acc,
            "elapsed_min": 0,
            "report": classification_report(
                test_labels, tta_3ens_preds, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(test_labels, tta_3ens_preds).tolist(),
        }
    )

    # ── r9_01: EffB0 + 320px + 300 epochs patience=60 ────────────────────────
    print("\n" + "=" * 70)
    print("  r9_01: EffB0 + 320px + 300 epochs (patience=60)")
    print("=" * 70)
    torch.manual_seed(SEED)
    m_r9_01 = build_effb0(dropout=0.3).to(device)
    h1, bva1, elt1 = train_model(
        m_r9_01,
        train_loader,
        val_loader,
        criterion,
        num_epochs=300,
        patience=60,
        save_path=os.path.join(MODEL_DIR, "r9_01_effb0_320px_300ep.pth"),
    )
    m_r9_01.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r9_01_effb0_320px_300ep.pth"), map_location=device
        )
    )
    _, ta1, tr1, pr1 = evaluate(m_r9_01, test_loader, criterion)
    print(f"\n  r9_01 Test (standard): {ta1:.4f}")
    print(classification_report(tr1, pr1, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(tr1, pr1))

    # TTA on r9_01
    print(f"\n  r9_01 TTA:")
    tta_probs_r9_01 = get_tta_probs(m_r9_01, test_paths, test_labels, tta_tfs)
    tta_preds_r9_01 = torch.argmax(tta_probs_r9_01, 1).numpy()
    tta_acc_r9_01 = accuracy_score(test_labels, tta_preds_r9_01)
    print(f"  r9_01 TTA: {tta_acc_r9_01:.4f}")

    # Ensemble r9_01 + r8_01
    print(f"\n  Ensemble r9_01 + r8_01:")
    std_probs_r8 = get_all_probs(m_r8_01, test_ds)
    std_probs_r9_01 = get_all_probs(m_r9_01, test_ds)
    ens2_probs = (std_probs_r8 + std_probs_r9_01) / 2
    ens2_preds = torch.argmax(ens2_probs, 1).numpy()
    ens2_acc = accuracy_score(test_labels, ens2_preds)
    print(f"  r9_01 + r8_01 ensemble: {ens2_acc:.4f}")
    print(
        classification_report(
            test_labels, ens2_preds, target_names=CLASS_NAMES, digits=4
        )
    )
    print(confusion_matrix(test_labels, ens2_preds))

    # 3-model ensemble: r7_01 + r8_01 + r9_01
    std_probs_r7 = get_all_probs(m_r7_01, test_ds)
    ens3_probs = (std_probs_r7 + std_probs_r8 + std_probs_r9_01) / 3
    ens3_preds = torch.argmax(ens3_probs, 1).numpy()
    ens3_acc = accuracy_score(test_labels, ens3_preds)
    print(f"  r7_01 + r8_01 + r9_01 ensemble: {ens3_acc:.4f}")

    results.extend(
        [
            {
                "name": "r9_01_effb0_320px_300ep",
                "best_val_acc": bva1,
                "test_acc": ta1,
                "elapsed_min": elt1,
                "report": classification_report(
                    tr1, pr1, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(tr1, pr1).tolist(),
            },
            {
                "name": "r9_01_tta",
                "best_val_acc": None,
                "test_acc": tta_acc_r9_01,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, tta_preds_r9_01, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, tta_preds_r9_01).tolist(),
            },
            {
                "name": "r9_01+r8_01_ensemble",
                "best_val_acc": None,
                "test_acc": ens2_acc,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, ens2_preds, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, ens2_preds).tolist(),
            },
            {
                "name": "r7+r8+r9_01_3ens",
                "best_val_acc": None,
                "test_acc": ens3_acc,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, ens3_preds, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, ens3_preds).tolist(),
            },
        ]
    )

    # ── r9_02: EffB0 + 320px + dropout=0.5 + 200ep ───────────────────────────
    print("\n" + "=" * 70)
    print("  r9_02: EffB0 + 320px + dropout=0.5 + 200 epochs (patience=40)")
    print("=" * 70)
    torch.manual_seed(SEED)
    m_r9_02 = build_effb0(dropout=0.5).to(device)
    h2, bva2, elt2 = train_model(
        m_r9_02,
        train_loader,
        val_loader,
        criterion,
        num_epochs=200,
        patience=40,
        save_path=os.path.join(MODEL_DIR, "r9_02_effb0_320px_do50_200ep.pth"),
    )
    m_r9_02.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r9_02_effb0_320px_do50_200ep.pth"),
            map_location=device,
        )
    )
    _, ta2, tr2, pr2 = evaluate(m_r9_02, test_loader, criterion)
    print(f"\n  r9_02 Test (standard): {ta2:.4f}")
    print(classification_report(tr2, pr2, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(tr2, pr2))

    # Ensemble r8_01 + r9_02
    std_probs_r9_02 = get_all_probs(m_r9_02, test_ds)
    ens_r8_r9_02_probs = (std_probs_r8 + std_probs_r9_02) / 2
    ens_r8_r9_02_preds = torch.argmax(ens_r8_r9_02_probs, 1).numpy()
    ens_r8_r9_02_acc = accuracy_score(test_labels, ens_r8_r9_02_preds)
    print(f"\n  r8_01 + r9_02 ensemble: {ens_r8_r9_02_acc:.4f}")

    # 3-model ensemble r7_01 + r8_01 + r9_02
    ens_3b_probs = (std_probs_r7 + std_probs_r8 + std_probs_r9_02) / 3
    ens_3b_preds = torch.argmax(ens_3b_probs, 1).numpy()
    ens_3b_acc = accuracy_score(test_labels, ens_3b_preds)
    print(f"  r7_01 + r8_01 + r9_02 ensemble: {ens_3b_acc:.4f}")

    # TTA on r9_02
    tta_probs_r9_02 = get_tta_probs(m_r9_02, test_paths, test_labels, tta_tfs)
    tta_preds_r9_02 = torch.argmax(tta_probs_r9_02, 1).numpy()
    tta_acc_r9_02 = accuracy_score(test_labels, tta_preds_r9_02)
    print(f"  r9_02 TTA: {tta_acc_r9_02:.4f}")

    results.extend(
        [
            {
                "name": "r9_02_effb0_320px_do50_200ep",
                "best_val_acc": bva2,
                "test_acc": ta2,
                "elapsed_min": elt2,
                "report": classification_report(
                    tr2, pr2, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(tr2, pr2).tolist(),
            },
            {
                "name": "r8_01+r9_02_ensemble",
                "best_val_acc": None,
                "test_acc": ens_r8_r9_02_acc,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, ens_r8_r9_02_preds, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, ens_r8_r9_02_preds).tolist(),
            },
            {
                "name": "r7+r8+r9_02_3ens",
                "best_val_acc": None,
                "test_acc": ens_3b_acc,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, ens_3b_preds, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, ens_3b_preds).tolist(),
            },
            {
                "name": "r9_02_tta",
                "best_val_acc": None,
                "test_acc": tta_acc_r9_02,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, tta_preds_r9_02, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, tta_preds_r9_02).tolist(),
            },
        ]
    )

    # ── r9_03: EfficientNetB2 + 320px + do=0.5 + 200ep + lr=1e-4 ─────────────
    print("\n" + "=" * 70)
    print("  r9_03: EfficientNetB2 + 320px + dropout=0.5 + 200ep + lr=1e-4")
    print("=" * 70)
    torch.manual_seed(SEED)
    m_r9_03 = build_effb2(dropout=0.5).to(device)
    h3, bva3, elt3 = train_model(
        m_r9_03,
        train_loader,
        val_loader,
        criterion,
        num_epochs=200,
        patience=40,
        save_path=os.path.join(MODEL_DIR, "r9_03_effb2_320px_do50_200ep.pth"),
        lr=1e-4,
    )
    m_r9_03.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r9_03_effb2_320px_do50_200ep.pth"),
            map_location=device,
        )
    )
    _, ta3, tr3, pr3 = evaluate(m_r9_03, test_loader, criterion)
    print(f"\n  r9_03 Test (standard): {ta3:.4f}")
    print(classification_report(tr3, pr3, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(tr3, pr3))

    # Ensemble r8_01 + r9_03 (EffB0 + EffB2 — architectural diversity)
    std_probs_r9_03 = get_all_probs(m_r9_03, test_ds)
    ens_b0_b2_probs = (std_probs_r8 + std_probs_r9_03) / 2
    ens_b0_b2_preds = torch.argmax(ens_b0_b2_probs, 1).numpy()
    ens_b0_b2_acc = accuracy_score(test_labels, ens_b0_b2_preds)
    print(f"  r8_01(B0) + r9_03(B2) ensemble: {ens_b0_b2_acc:.4f}")

    # 3-model: r7_01 + r8_01 + r9_03
    ens_3c_probs = (std_probs_r7 + std_probs_r8 + std_probs_r9_03) / 3
    ens_3c_preds = torch.argmax(ens_3c_probs, 1).numpy()
    ens_3c_acc = accuracy_score(test_labels, ens_3c_preds)
    print(f"  r7_01 + r8_01 + r9_03 ensemble: {ens_3c_acc:.4f}")

    # TTA on r9_03
    tta_probs_r9_03 = get_tta_probs(m_r9_03, test_paths, test_labels, tta_tfs)
    tta_preds_r9_03 = torch.argmax(tta_probs_r9_03, 1).numpy()
    tta_acc_r9_03 = accuracy_score(test_labels, tta_preds_r9_03)
    print(f"  r9_03 TTA: {tta_acc_r9_03:.4f}")

    # Grand ensemble: r7_01 + r8_01 + r9_01 + r9_03 (if we have r9_01 probs)
    if "tta_probs_r9_01" in dir():
        grand_probs = (
            std_probs_r7 + std_probs_r8 + std_probs_r9_01 + std_probs_r9_03
        ) / 4
        grand_preds = torch.argmax(grand_probs, 1).numpy()
        grand_acc = accuracy_score(test_labels, grand_preds)
        print(f"  GRAND 4-model ensemble (r7+r8+r9_01+r9_03): {grand_acc:.4f}")
        results.append(
            {
                "name": "r9_grand_4ens_r7_r8_r9_01_r9_03",
                "best_val_acc": None,
                "test_acc": grand_acc,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, grand_preds, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, grand_preds).tolist(),
            }
        )

    results.extend(
        [
            {
                "name": "r9_03_effb2_320px_do50_200ep",
                "best_val_acc": bva3,
                "test_acc": ta3,
                "elapsed_min": elt3,
                "report": classification_report(
                    tr3, pr3, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(tr3, pr3).tolist(),
            },
            {
                "name": "r8_01+r9_03_B0B2_ensemble",
                "best_val_acc": None,
                "test_acc": ens_b0_b2_acc,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, ens_b0_b2_preds, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, ens_b0_b2_preds).tolist(),
            },
            {
                "name": "r7+r8+r9_03_3ens",
                "best_val_acc": None,
                "test_acc": ens_3c_acc,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, ens_3c_preds, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, ens_3c_preds).tolist(),
            },
            {
                "name": "r9_03_tta",
                "best_val_acc": None,
                "test_acc": tta_acc_r9_03,
                "elapsed_min": 0,
                "report": classification_report(
                    test_labels, tta_preds_r9_03, target_names=CLASS_NAMES, digits=4
                ),
                "cm": confusion_matrix(test_labels, tta_preds_r9_03).tolist(),
            },
        ]
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("ROUND 9 SUMMARY")
    print("=" * 70)
    print(f"  {'Name':<42}  {'ValAcc':>7}  {'TestAcc':>8}  {'Time(m)':>8}")
    print("  " + "-" * 75)
    for r in sorted(results, key=lambda x: x["test_acc"], reverse=True):
        va = f"{r['best_val_acc']:.4f}" if r["best_val_acc"] is not None else "  N/A "
        print(
            f"  {r['name']:<42}  {va:>7}  {r['test_acc']:>8.4f}  {r['elapsed_min']:>8.1f}"
        )

    with open("train_round9_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to train_round9_results.json")


if __name__ == "__main__":
    main()

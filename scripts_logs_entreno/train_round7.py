"""
train_round7.py

Round 7: Push beyond 0.6433.

Experiments:
  r7_01 — EffB0 + 320px crop + 120 epochs / patience=30
           320px model early-stopped at ep8 in R6 — needs much more training.
  r7_02 — Ensemble of r6_01 (224px,80ep) + r6_04 (320px,23ep)
           Complementary error profiles: r6_01 precision-oriented, r6_04 recall-oriented.
  r7_03 — EffB0 + CosineAnnealingWarmRestarts (T_0=10, T_mult=2) vs OneCycleLR
           Restarts can escape local minima; may converge to a better solution.
  r7_04 — EfficientNetB2 + 320px + 80 epochs (B2 + bigger crop, first try)
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


def make_transforms(crop_size=224):
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


def build_effb2(dropout=0.3):
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


def evaluate_ensemble(models_list, loader):
    for m in models_list:
        m.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            prob_sum = sum(F.softmax(m(imgs), dim=1) for m in models_list)
            preds.extend(torch.argmax(prob_sum, 1).cpu().numpy())
            trues.extend(labels.numpy())
    return accuracy_score(trues, preds), trues, preds


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
    scheduler_type="onecycle",
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        step_per_batch = True
    elif scheduler_type == "cosine_warm_restarts":
        # T_0=10 epochs, T_mult=2 → restarts at ep10, ep30, ep70...
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10 * len(train_loader),
            T_mult=2,
            eta_min=1e-6,
        )
        step_per_batch = True
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

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
            if step_per_batch:
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
    # Load datasets
    print("Loading datasets...")
    train_tf_224, val_tf_224 = make_transforms(224)
    train_tf_320, val_tf_320 = make_transforms(320)

    train_ds_224 = SplitImageDataset(os.path.join(DATA_DIR, "train"), train_tf_224)
    val_ds_224 = SplitImageDataset(os.path.join(DATA_DIR, "val"), val_tf_224)
    test_ds_224 = SplitImageDataset(os.path.join(DATA_DIR, "test"), val_tf_224)

    train_ds_320 = SplitImageDataset(os.path.join(DATA_DIR, "train"), train_tf_320)
    val_ds_320 = SplitImageDataset(os.path.join(DATA_DIR, "val"), val_tf_320)
    test_ds_320 = SplitImageDataset(os.path.join(DATA_DIR, "test"), val_tf_320)

    # Class weights (for 224 and 320 — same train split)
    lc = Counter(train_ds_224.labels)
    total = len(train_ds_224.labels)
    cw = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        cw[i] = total / (NUM_CLASSES * lc.get(i, 1))
    criterion_224 = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=0.1)
    criterion_320 = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=0.1)

    results = []

    # ── r7_01: EffB0 320px 120 epochs ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  r7_01: EffB0 + 320px crop + 120 epochs (patience=30)")
    print("=" * 70)
    train_loader = DataLoader(train_ds_320, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds_320, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds_320, batch_size=16, shuffle=False, num_workers=0)

    m = build_effb0(dropout=0.3).to(device)
    hist, bva, elt = train_model(
        m,
        train_loader,
        val_loader,
        criterion_320,
        num_epochs=120,
        patience=30,
        save_path=os.path.join(MODEL_DIR, "r7_01_effb0_320px_120ep.pth"),
        lr=3e-4,
        scheduler_type="onecycle",
    )
    m.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r7_01_effb0_320px_120ep.pth"), map_location=device
        )
    )
    _, test_acc, trues, preds = evaluate(m, test_loader, criterion_320)
    print(f"  Test: {test_acc:.4f}")
    print(classification_report(trues, preds, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(trues, preds))
    results.append(
        {
            "name": "r7_01_effb0_320px_120ep",
            "best_val_acc": bva,
            "test_acc": test_acc,
            "elapsed_min": elt,
            "report": classification_report(
                trues, preds, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(trues, preds).tolist(),
        }
    )

    # ── r7_02: Ensemble r6_01 + r6_04 ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  r7_02: Ensemble r6_01 (224px,80ep) + r6_04 (320px,early-stop)")
    print("=" * 70)
    m_r6_01 = build_effb0(dropout=0.3).to(device)
    m_r6_01.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "r6_01_effb0_80ep.pth"), map_location=device)
    )

    m_r6_04 = build_effb0(dropout=0.3).to(device)
    m_r6_04.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r6_04_effb0_320px.pth"), map_location=device
        )
    )

    # For ensemble, we need compatible loaders. Use 224 test (r6_01's resolution):
    test_loader_224 = DataLoader(
        test_ds_224, batch_size=32, shuffle=False, num_workers=0
    )
    test_loader_320 = DataLoader(
        test_ds_320, batch_size=16, shuffle=False, num_workers=0
    )

    # Run separately and combine logits (need aligned batches — process image-by-image)
    # Simple approach: get all logits for both, then average
    def get_all_probs(model, ds):
        model.eval()
        probs_list = []
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        with torch.no_grad():
            for imgs, _ in loader:
                out = model(imgs.to(device))
                probs_list.append(F.softmax(out, dim=1).cpu())
        return torch.cat(probs_list, dim=0)

    probs_r6_01 = get_all_probs(m_r6_01, test_ds_224)
    probs_r6_04 = get_all_probs(m_r6_04, test_ds_320)

    # Average
    ens_probs = (probs_r6_01 + probs_r6_04) / 2
    ens_preds = torch.argmax(ens_probs, dim=1).numpy()
    ens_trues = test_ds_224.labels  # same images, same order
    ens_acc = accuracy_score(ens_trues, ens_preds)
    print(f"  Ensemble Test Accuracy: {ens_acc:.4f}")
    print(
        classification_report(ens_trues, ens_preds, target_names=CLASS_NAMES, digits=4)
    )
    print(confusion_matrix(ens_trues, ens_preds))
    results.append(
        {
            "name": "r7_02_ensemble_r6_01_r6_04",
            "best_val_acc": None,
            "test_acc": ens_acc,
            "elapsed_min": 0,
            "report": classification_report(
                ens_trues, ens_preds, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(ens_trues, ens_preds).tolist(),
        }
    )

    # ── r7_03: EffB0 224px CosineWarmRestarts ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  r7_03: EffB0 224px + CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    print("=" * 70)
    train_loader_224 = DataLoader(
        train_ds_224, batch_size=32, shuffle=True, num_workers=0
    )
    val_loader_224 = DataLoader(val_ds_224, batch_size=32, shuffle=False, num_workers=0)
    test_loader_224b = DataLoader(
        test_ds_224, batch_size=32, shuffle=False, num_workers=0
    )

    m3 = build_effb0(dropout=0.3).to(device)
    hist3, bva3, elt3 = train_model(
        m3,
        train_loader_224,
        val_loader_224,
        criterion_224,
        num_epochs=80,
        patience=20,
        save_path=os.path.join(MODEL_DIR, "r7_03_effb0_coswr.pth"),
        lr=3e-4,
        scheduler_type="cosine_warm_restarts",
    )
    m3.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r7_03_effb0_coswr.pth"), map_location=device
        )
    )
    _, test_acc3, trues3, preds3 = evaluate(m3, test_loader_224b, criterion_224)
    print(f"  Test: {test_acc3:.4f}")
    print(classification_report(trues3, preds3, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(trues3, preds3))
    results.append(
        {
            "name": "r7_03_effb0_coswr",
            "best_val_acc": bva3,
            "test_acc": test_acc3,
            "elapsed_min": elt3,
            "report": classification_report(
                trues3, preds3, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(trues3, preds3).tolist(),
        }
    )

    # ── r7_04: EfficientNetB2 + 320px + 80ep ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  r7_04: EfficientNetB2 + 320px crop + 80 epochs (patience=20)")
    print("=" * 70)
    train_loader_320 = DataLoader(
        train_ds_320, batch_size=16, shuffle=True, num_workers=0
    )
    val_loader_320 = DataLoader(val_ds_320, batch_size=16, shuffle=False, num_workers=0)
    test_loader_320b = DataLoader(
        test_ds_320, batch_size=16, shuffle=False, num_workers=0
    )

    m4 = build_effb2(dropout=0.3).to(device)
    hist4, bva4, elt4 = train_model(
        m4,
        train_loader_320,
        val_loader_320,
        criterion_320,
        num_epochs=80,
        patience=20,
        save_path=os.path.join(MODEL_DIR, "r7_04_effb2_320px.pth"),
        lr=3e-4,
        scheduler_type="onecycle",
    )
    m4.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "r7_04_effb2_320px.pth"), map_location=device
        )
    )
    _, test_acc4, trues4, preds4 = evaluate(m4, test_loader_320b, criterion_320)
    print(f"  Test: {test_acc4:.4f}")
    print(classification_report(trues4, preds4, target_names=CLASS_NAMES, digits=4))
    print(confusion_matrix(trues4, preds4))
    results.append(
        {
            "name": "r7_04_effb2_320px",
            "best_val_acc": bva4,
            "test_acc": test_acc4,
            "elapsed_min": elt4,
            "report": classification_report(
                trues4, preds4, target_names=CLASS_NAMES, digits=4
            ),
            "cm": confusion_matrix(trues4, preds4).tolist(),
        }
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("ROUND 7 SUMMARY")
    print("=" * 70)
    print(f"  {'Name':<35}  {'ValAcc':>7}  {'TestAcc':>8}  {'Time(m)':>8}")
    print("  " + "-" * 65)
    for r in sorted(results, key=lambda x: x["test_acc"], reverse=True):
        va = f"{r['best_val_acc']:.4f}" if r["best_val_acc"] is not None else "  N/A "
        print(
            f"  {r['name']:<35}  {va:>7}  {r['test_acc']:>8.4f}  {r['elapsed_min']:>8.1f}"
        )

    with open("train_round7_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to train_round7_results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment runner for diabetic retinopathy classification.
Tests multiple model architectures, optimizers, and hyperparameter configs.
Results are documented in experiments.txt.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report
import random, os, time
import numpy as np
from PIL import Image
from typing import Optional
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

DATA_SPLIT_DIR = "data_split"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
CLASS_NAMES = ["0-noDR", "1-mild", "2-moderate", "3-severe", "4-proliferativeDR"]
NUM_CLASSES = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SplitImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
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


def get_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        weights[i] = total / (num_classes * counts.get(i, 1))
    return weights.to(device)


def get_transforms(img_size=224, augment=True):
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.1),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


def build_model(arch: str, num_classes: int, dropout: float = 0.4) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feat = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_feat = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_feat, num_classes)
        )
    elif arch == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_feat, num_classes)
        )
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_feat = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, true_labels = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            total_loss += criterion(out, labels).item()
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return (
        total_loss / len(loader),
        accuracy_score(true_labels, preds),
        preds,
        true_labels,
    )


def run_experiment(cfg: dict) -> dict:
    name = cfg["name"]
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 70}")

    # Datasets
    img_size = cfg.get("img_size", 224)
    train_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "train"), get_transforms(img_size, augment=True)
    )
    val_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "val"), get_transforms(img_size, augment=False)
    )
    test_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "test"), get_transforms(img_size, augment=False)
    )

    bs = cfg.get("batch_size", 16)
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=0)

    # Model
    model = build_model(cfg["arch"], NUM_CLASSES, cfg.get("dropout", 0.4)).to(device)

    # Loss
    cw = get_class_weights(train_ds.labels, NUM_CLASSES)
    label_smoothing = cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    # Optimizer
    lr = cfg.get("lr", 1e-4)
    wd = cfg.get("weight_decay", 1e-4)
    opt_name = cfg.get("optimizer", "adamw")
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True
        )
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
        )

    # Scheduler
    num_epochs = cfg.get("num_epochs", 60)
    sched_name = cfg.get("scheduler", "onecycle")
    if sched_name == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,
        )
        step_per_batch = True
    elif sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        step_per_batch = False
    elif sched_name == "reduce":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=5, min_lr=1e-7
        )
        step_per_batch = False
    elif sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        step_per_batch = False

    # Training
    best_val_acc = 0.0
    patience = cfg.get("patience", 15)
    patience_counter = 0
    best_path = os.path.join(MODEL_DIR, f"best_{name.replace(' ', '_')}.pth")
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step_per_batch:
                scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        if not step_per_batch:
            if sched_name == "reduce":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"  Early stop @ epoch {epoch + 1}, best val_acc={best_val_acc:.4f}"
                )
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{num_epochs}: train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

    elapsed = time.time() - start

    # Test evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_preds, test_true = evaluate(model, test_loader, criterion)

    report = classification_report(
        test_true, test_preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    print(f"\nResults for {name}:")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Test Accuracy:     {test_acc:.4f}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print(report)

    return {
        "name": name,
        "arch": cfg["arch"],
        "optimizer": opt_name,
        "scheduler": sched_name,
        "lr": lr,
        "dropout": cfg.get("dropout", 0.4),
        "label_smoothing": label_smoothing,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "elapsed_min": elapsed / 60,
        "report": report,
        "best_path": best_path,
    }


# ============================================================
# Experiments
# ============================================================

experiments = [
    # Baseline: ResNet18 + AdamW + OneCycle (already run, re-running for consistency)
    {
        "name": "exp1_resnet18_adamw_onecycle",
        "arch": "resnet18",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "batch_size": 16,
        "num_epochs": 60,
        "patience": 15,
        "label_smoothing": 0.0,
    },
    # Exp2: ResNet18 + label smoothing
    {
        "name": "exp2_resnet18_label_smoothing",
        "arch": "resnet18",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "batch_size": 16,
        "num_epochs": 60,
        "patience": 15,
        "label_smoothing": 0.1,
    },
    # Exp3: ResNet18 + SGD + cosine
    {
        "name": "exp3_resnet18_sgd_cosine",
        "arch": "resnet18",
        "optimizer": "sgd",
        "scheduler": "cosine",
        "lr": 1e-2,
        "weight_decay": 5e-4,
        "dropout": 0.4,
        "batch_size": 16,
        "num_epochs": 80,
        "patience": 20,
        "label_smoothing": 0.0,
    },
    # Exp4: EfficientNetB0 + AdamW + OneCycle
    {
        "name": "exp4_efficientnetb0_adamw_onecycle",
        "arch": "efficientnet_b0",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "batch_size": 16,
        "num_epochs": 60,
        "patience": 15,
        "label_smoothing": 0.0,
    },
    # Exp5: EfficientNetB0 + label smoothing + higher LR
    {
        "name": "exp5_efficientnetb0_smoothing",
        "arch": "efficientnet_b0",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 16,
        "num_epochs": 60,
        "patience": 15,
        "label_smoothing": 0.1,
    },
    # Exp6: ResNet50 + AdamW + OneCycle
    {
        "name": "exp6_resnet50_adamw_onecycle",
        "arch": "resnet50",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "dropout": 0.5,
        "batch_size": 16,
        "num_epochs": 60,
        "patience": 15,
        "label_smoothing": 0.1,
    },
    # Exp7: ResNet18 + AdamW + ReduceLROnPlateau (longer training)
    {
        "name": "exp7_resnet18_adamw_reduce",
        "arch": "resnet18",
        "optimizer": "adamw",
        "scheduler": "reduce",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "batch_size": 16,
        "num_epochs": 100,
        "patience": 25,
        "label_smoothing": 0.1,
    },
    # Exp8: EfficientNetB2 + AdamW + OneCycle (larger model)
    {
        "name": "exp8_efficientnetb2_adamw_onecycle",
        "arch": "efficientnet_b2",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "batch_size": 16,
        "num_epochs": 60,
        "patience": 15,
        "label_smoothing": 0.1,
        "img_size": 260,
    },
    # Exp9: MobileNetV3 (lightweight, fast) + AdamW
    {
        "name": "exp9_mobilenetv3_adamw",
        "arch": "mobilenet_v3_small",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 32,
        "num_epochs": 60,
        "patience": 15,
        "label_smoothing": 0.1,
    },
    # Exp10: Best arch with best settings found above - deeper fine-tuning
    {
        "name": "exp10_efficientnetb0_best_settings",
        "arch": "efficientnet_b0",
        "optimizer": "adamw",
        "scheduler": "cosine",
        "lr": 1e-4,
        "weight_decay": 5e-4,
        "dropout": 0.35,
        "batch_size": 16,
        "num_epochs": 100,
        "patience": 25,
        "label_smoothing": 0.1,
    },
]

all_results = []
for cfg in experiments:
    try:
        result = run_experiment(cfg)
        all_results.append(result)
    except Exception as e:
        print(f"ERROR in {cfg['name']}: {e}")
        import traceback

        traceback.print_exc()

# Sort by test accuracy
all_results.sort(key=lambda x: x["test_acc"], reverse=True)

print("\n" + "=" * 70)
print("FINAL RANKINGS BY TEST ACCURACY")
print("=" * 70)
print(f"{'Rank':>4} {'Name':>45} {'Val Acc':>9} {'Test Acc':>9} {'Time(m)':>8}")
print("-" * 80)
for i, r in enumerate(all_results):
    print(
        f"{i + 1:>4} {r['name']:>45} {r['best_val_acc']:>9.4f} {r['test_acc']:>9.4f} {r['elapsed_min']:>8.1f}"
    )

# Write results to file
results_text = []
results_text.append("=" * 70)
results_text.append("DIABETIC RETINOPATHY CLASSIFICATION - EXPERIMENT RESULTS")
results_text.append("=" * 70)
results_text.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
results_text.append(f"Device: {device}")
results_text.append(f"Dataset: data_split (train=848, val=241, test=127)")
results_text.append(f"Classes: {CLASS_NAMES}")
results_text.append("")
results_text.append("RANKINGS BY TEST ACCURACY:")
results_text.append(
    f"{'Rank':>4} {'Name':>45} {'Val Acc':>9} {'Test Acc':>9} {'Time(m)':>8}"
)
results_text.append("-" * 80)
for i, r in enumerate(all_results):
    results_text.append(
        f"{i + 1:>4} {r['name']:>45} {r['best_val_acc']:>9.4f} {r['test_acc']:>9.4f} {r['elapsed_min']:>8.1f}"
    )

results_text.append("")
results_text.append("=" * 70)
results_text.append("DETAILED RESULTS")
results_text.append("=" * 70)
for r in all_results:
    results_text.append(f"\n--- {r['name']} ---")
    results_text.append(f"Architecture:    {r['arch']}")
    results_text.append(f"Optimizer:       {r['optimizer']}  lr={r['lr']}")
    results_text.append(f"Scheduler:       {r['scheduler']}")
    results_text.append(f"Dropout:         {r['dropout']}")
    results_text.append(f"Label smoothing: {r['label_smoothing']}")
    results_text.append(f"Best Val Acc:    {r['best_val_acc']:.4f}")
    results_text.append(f"Test Accuracy:   {r['test_acc']:.4f}")
    results_text.append(f"Training time:   {r['elapsed_min']:.1f} min")
    results_text.append(f"Best model:      {r['best_path']}")
    results_text.append("Classification Report:")
    results_text.append(r["report"])

with open("experiments.txt", "w") as f:
    f.write("\n".join(results_text))

print(f"\nResults saved to experiments.txt")
print(
    f"Best model: {all_results[0]['name']} with test_acc={all_results[0]['test_acc']:.4f}"
)
print(f"Best model path: {all_results[0]['best_path']}")

#!/usr/bin/env python3
"""
Round 2 experiments: focused on EfficientNetB0 (best arch) with varied configs.
Also tries layer-freezing strategies and ensemble ideas.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


def get_weighted_sampler(labels, num_classes):
    """Create WeightedRandomSampler to oversample minority classes."""
    counts = Counter(labels)
    sample_weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def get_transforms(img_size=224, augment=True, strong_aug=False):
    if augment and strong_aug:
        return transforms.Compose(
            [
                transforms.Resize((img_size + 64, img_size + 64)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(45),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            ]
        )
    elif augment:
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


def build_efficientnet_b0(num_classes, dropout=0.4, freeze_backbone=False):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    if freeze_backbone:
        for param in m.features.parameters():
            param.requires_grad = False
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.5),
        nn.Linear(256, num_classes),
    )
    return m


def build_resnet18(num_classes, dropout=0.4, freeze_backbone=False):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for name, param in m.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.5),
        nn.Linear(256, num_classes),
    )
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


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    patience,
    name,
    step_per_batch=True,
):
    best_val_acc = 0.0
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
            if hasattr(scheduler, "step"):
                try:
                    scheduler.step(val_loss)
                except TypeError:
                    scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}: train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

    elapsed = time.time() - start
    return best_val_acc, best_path, elapsed


def run_experiment(cfg):
    name = cfg["name"]
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 70}")

    img_size = cfg.get("img_size", 224)
    strong_aug = cfg.get("strong_aug", False)
    use_sampler = cfg.get("weighted_sampler", False)

    train_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "train"),
        get_transforms(img_size, True, strong_aug),
    )
    val_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "val"), get_transforms(img_size, False)
    )
    test_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "test"), get_transforms(img_size, False)
    )

    bs = cfg.get("batch_size", 16)
    if use_sampler:
        sampler = get_weighted_sampler(train_ds.labels, NUM_CLASSES)
        train_loader = DataLoader(train_ds, bs, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=0)

    # Model
    arch = cfg.get("arch", "efficientnet_b0")
    freeze = cfg.get("freeze_backbone", False)
    dropout = cfg.get("dropout", 0.4)
    if arch == "efficientnet_b0":
        model = build_efficientnet_b0(NUM_CLASSES, dropout, freeze).to(device)
    elif arch == "resnet18":
        model = build_resnet18(NUM_CLASSES, dropout, freeze).to(device)

    # Loss
    cw = get_class_weights(train_ds.labels, NUM_CLASSES)
    label_smoothing = cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    # Optimizer - support differential LR for backbone vs head
    lr = cfg.get("lr", 3e-4)
    wd = cfg.get("weight_decay", 1e-4)
    diff_lr = cfg.get("diff_lr", False)
    opt_name = cfg.get("optimizer", "adamw")

    if diff_lr and arch == "efficientnet_b0":
        backbone_params = [
            p
            for n, p in model.named_parameters()
            if "classifier" not in n and p.requires_grad
        ]
        head_params = [
            p
            for n, p in model.named_parameters()
            if "classifier" in n and p.requires_grad
        ]
        param_groups = [
            {"params": backbone_params, "lr": lr / 10},
            {"params": head_params, "lr": lr},
        ]
    elif diff_lr and arch == "resnet18":
        backbone_params = [
            p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad
        ]
        head_params = [
            p for n, p in model.named_parameters() if "fc" in n and p.requires_grad
        ]
        param_groups = [
            {"params": backbone_params, "lr": lr / 10},
            {"params": head_params, "lr": lr},
        ]
    else:
        param_groups = model.parameters()

    if opt_name == "adamw":
        optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = optim.Adam(param_groups, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            param_groups, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True
        )

    num_epochs = cfg.get("num_epochs", 80)
    patience = cfg.get("patience", 20)
    sched_name = cfg.get("scheduler", "onecycle")

    if sched_name == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        step_per_batch = True
    elif sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-7
        )
        step_per_batch = False
    elif sched_name == "reduce":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=5
        )
        step_per_batch = False

    best_val_acc, best_path, elapsed = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        patience,
        name,
        step_per_batch,
    )

    # Test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_preds, test_true = evaluate(model, test_loader, criterion)
    report = classification_report(
        test_true, test_preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    print(
        f"\nResults: val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  time={elapsed / 60:.1f}m"
    )
    print(report)

    return {
        "name": name,
        "arch": arch,
        "optimizer": opt_name,
        "scheduler": sched_name,
        "lr": lr,
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "weighted_sampler": use_sampler,
        "diff_lr": diff_lr,
        "strong_aug": strong_aug,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "elapsed_min": elapsed / 60,
        "report": report,
        "best_path": best_path,
    }


# ============================================================
# Round 2 Experiments - focused refinement
# ============================================================
experiments_r2 = [
    # R2-1: Best from R1 (EffB0 + smoothing + lr=3e-4) with weighted sampler
    {
        "name": "r2_01_effb0_smoothing_weighted_sampler",
        "arch": "efficientnet_b0",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 16,
        "num_epochs": 80,
        "patience": 20,
        "label_smoothing": 0.1,
        "weighted_sampler": True,
    },
    # R2-2: Differential LR (lower for backbone, higher for head)
    {
        "name": "r2_02_effb0_diff_lr",
        "arch": "efficientnet_b0",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 16,
        "num_epochs": 80,
        "patience": 20,
        "label_smoothing": 0.1,
        "diff_lr": True,
    },
    # R2-3: Strong augmentation + weighted sampler
    {
        "name": "r2_03_effb0_strong_aug_sampler",
        "arch": "efficientnet_b0",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 16,
        "num_epochs": 80,
        "patience": 20,
        "label_smoothing": 0.1,
        "strong_aug": True,
        "weighted_sampler": True,
    },
    # R2-4: ResNet18 + diff LR + weighted sampler + strong aug
    {
        "name": "r2_04_resnet18_diff_lr_sampler",
        "arch": "resnet18",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "dropout": 0.4,
        "batch_size": 16,
        "num_epochs": 80,
        "patience": 20,
        "label_smoothing": 0.1,
        "diff_lr": True,
        "weighted_sampler": True,
    },
    # R2-5: EffB0 + cosine + longer training + smoothing 0.05
    {
        "name": "r2_05_effb0_cosine_longer",
        "arch": "efficientnet_b0",
        "optimizer": "adamw",
        "scheduler": "cosine",
        "lr": 2e-4,
        "weight_decay": 2e-4,
        "dropout": 0.35,
        "batch_size": 16,
        "num_epochs": 120,
        "patience": 30,
        "label_smoothing": 0.05,
        "weighted_sampler": True,
    },
    # R2-6: SGD nesterov on EffB0 (sometimes beats AdamW at fine-tuning)
    {
        "name": "r2_06_effb0_sgd_nesterov",
        "arch": "efficientnet_b0",
        "optimizer": "sgd",
        "scheduler": "cosine",
        "lr": 1e-2,
        "weight_decay": 5e-4,
        "dropout": 0.3,
        "batch_size": 16,
        "num_epochs": 100,
        "patience": 25,
        "label_smoothing": 0.1,
        "weighted_sampler": True,
    },
    # R2-7: Two-phase: freeze backbone first, then unfreeze
    {
        "name": "r2_07_effb0_two_phase",
        "arch": "efficientnet_b0",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "batch_size": 16,
        "num_epochs": 80,
        "patience": 20,
        "label_smoothing": 0.1,
        "weighted_sampler": True,
        "freeze_backbone": False,  # handled custom below
        "_two_phase": True,
    },
]


def run_two_phase(cfg):
    """Phase 1: Freeze backbone, train head. Phase 2: Unfreeze all, fine-tune."""
    name = cfg["name"]
    print(f"\n{'=' * 70}")
    print(f"TWO-PHASE EXPERIMENT: {name}")
    print(f"{'=' * 70}")

    img_size = cfg.get("img_size", 224)
    use_sampler = cfg.get("weighted_sampler", False)
    train_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "train"), get_transforms(img_size, True)
    )
    val_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "val"), get_transforms(img_size, False)
    )
    test_ds = SplitImageDataset(
        os.path.join(DATA_SPLIT_DIR, "test"), get_transforms(img_size, False)
    )

    bs = cfg.get("batch_size", 16)
    if use_sampler:
        sampler = get_weighted_sampler(train_ds.labels, NUM_CLASSES)
        train_loader = DataLoader(train_ds, bs, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=0)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    model = model.to(device)

    cw = get_class_weights(train_ds.labels, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    # Phase 1: head only, 20 epochs
    print("  Phase 1: Training head only (frozen backbone)...")
    optimizer1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler1 = optim.lr_scheduler.OneCycleLR(
        optimizer1,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=20,
        pct_start=0.1,
    )
    best_val_acc1, best_path1, elapsed1 = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer1,
        scheduler1,
        20,
        20,
        name + "_phase1",
        True,
    )
    print(f"  Phase 1 best val_acc: {best_val_acc1:.4f}")

    # Phase 2: unfreeze all
    print("  Phase 2: Fine-tuning all layers...")
    model.load_state_dict(torch.load(best_path1, map_location=device))
    for param in model.parameters():
        param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer2 = optim.AdamW(
        [{"params": backbone_params, "lr": 3e-5}, {"params": head_params, "lr": 3e-4}],
        weight_decay=1e-4,
    )
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=60, eta_min=1e-7
    )
    best_path2 = os.path.join(MODEL_DIR, f"best_{name.replace(' ', '_')}.pth")

    best_val_acc2 = 0.0
    patience_counter = 0
    start2 = time.time()
    for epoch in range(60):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer2.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()
            total_loss += loss.item()
        scheduler2.step()
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        if val_acc > best_val_acc2:
            best_val_acc2 = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path2)
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Phase2 Epoch {epoch + 1:3d}: train_loss={avg_train_loss:.4f}  val_acc={val_acc:.4f}"
            )

    elapsed2 = time.time() - start2
    elapsed = elapsed1 + elapsed2

    model.load_state_dict(torch.load(best_path2, map_location=device))
    test_loss, test_acc, test_preds, test_true = evaluate(model, test_loader, criterion)
    report = classification_report(
        test_true, test_preds, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    print(
        f"\nResults: val_acc={best_val_acc2:.4f}  test_acc={test_acc:.4f}  time={elapsed / 60:.1f}m"
    )
    print(report)

    return {
        "name": name,
        "arch": "efficientnet_b0_two_phase",
        "best_val_acc": best_val_acc2,
        "test_acc": test_acc,
        "elapsed_min": elapsed / 60,
        "report": report,
        "best_path": best_path2,
        "label_smoothing": 0.1,
        "optimizer": "adamw_two_phase",
        "scheduler": "onecycle+cosine",
        "lr": "1e-3->3e-5/3e-4",
        "dropout": 0.4,
        "weighted_sampler": True,
        "diff_lr": True,
    }


all_results = []
for cfg in experiments_r2:
    try:
        if cfg.get("_two_phase"):
            result = run_two_phase(cfg)
        else:
            result = run_experiment(cfg)
        all_results.append(result)
    except Exception as e:
        print(f"ERROR in {cfg['name']}: {e}")
        import traceback

        traceback.print_exc()

# Sort by test accuracy
all_results.sort(key=lambda x: x["test_acc"], reverse=True)

print("\n" + "=" * 70)
print("ROUND 2 RANKINGS BY TEST ACCURACY")
print("=" * 70)
print(f"{'Rank':>4} {'Name':>45} {'Val Acc':>9} {'Test Acc':>9}")
print("-" * 65)
for i, r in enumerate(all_results):
    print(f"{i + 1:>4} {r['name']:>45} {r['best_val_acc']:>9.4f} {r['test_acc']:>9.4f}")

print(f"\nBest: {all_results[0]['name']}  test_acc={all_results[0]['test_acc']:.4f}")
print(f"Best model path: {all_results[0]['best_path']}")

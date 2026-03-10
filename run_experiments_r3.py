#!/usr/bin/env python3
"""
Round 3 experiments:
1. TTA (Test-Time Augmentation) on best R2 model
2. Ensemble of top models (R1 + R2)
3. EffB0 + WeightedSampler + tuned lr
4. EffB1 + WeightedSampler
5. Mixup augmentation + WeightedSampler
6. Focal loss + WeightedSampler
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
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
np.random.seed(SEED)

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


def get_val_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_tta_transform(img_size=224):
    """Single augmented transform for TTA."""
    return transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_train_transform(img_size=224):
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


def get_class_weights(labels):
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        weights[i] = total / (NUM_CLASSES * counts.get(i, 1))
    return weights.to(device)


def get_weighted_sampler(labels):
    counts = Counter(labels)
    sample_weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def build_effb0(dropout=0.3, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, NUM_CLASSES),
    )
    return m


def build_effb0_wide(dropout=0.3, pretrained=True):
    """EffB0 with same expanded head as R2 experiments."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    return m


def build_effb1(dropout=0.3, pretrained=True):
    weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b1(weights=weights)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, NUM_CLASSES),
    )
    return m


def build_resnet18(dropout=0.4, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    return m


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
    return acc, report, preds, trues


def evaluate_tta(model, dataset, n_aug=8, img_size=224):
    """Test-Time Augmentation: average predictions over n_aug augmented views."""
    model.eval()
    tta_transform = get_tta_transform(img_size)
    preds_all, trues = [], []

    for idx in range(len(dataset)):
        true_label = dataset.labels[idx]
        trues.append(true_label)
        # Original + augmented predictions
        with Image.open(dataset.image_paths[idx]) as raw_img:
            raw_img = raw_img.convert("RGB")

        logits_sum = None
        # 1 original + n_aug augmented
        val_t = get_val_transform(img_size)
        imgs_list = [val_t(raw_img)] + [tta_transform(raw_img) for _ in range(n_aug)]
        imgs_tensor = torch.stack(imgs_list).to(device)

        with torch.no_grad():
            out = model(imgs_tensor)
            logits_sum = out.mean(0)

        preds_all.append(torch.argmax(logits_sum).item())

    acc = accuracy_score(trues, preds_all)
    report = classification_report(
        trues, preds_all, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    return acc, report


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, step_per_batch, use_mixup=False
):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)
            out = model(imgs)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            out = model(imgs)
            loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step_per_batch:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


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
    use_mixup=False,
):
    best_val_acc = 0.0
    patience_counter = 0
    best_path = os.path.join(MODEL_DIR, f"best_{name}.pth")
    start = time.time()

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            step_per_batch,
            use_mixup,
        )
        val_acc, _, _, _ = evaluate(model, val_loader)

        if not step_per_batch:
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
                f"  Epoch {epoch + 1:3d}: loss={avg_loss:.4f}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}"
            )

    elapsed = time.time() - start
    return best_val_acc, best_path, elapsed


# ============================================================
# Load datasets once
# ============================================================
train_ds_base = SplitImageDataset(
    os.path.join(DATA_SPLIT_DIR, "train"), get_train_transform()
)
val_ds = SplitImageDataset(os.path.join(DATA_SPLIT_DIR, "val"), get_val_transform())
test_ds = SplitImageDataset(os.path.join(DATA_SPLIT_DIR, "test"), get_val_transform())

val_loader = DataLoader(val_ds, 32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, 32, shuffle=False, num_workers=0)

cw = get_class_weights(train_ds_base.labels)
sampler = get_weighted_sampler(train_ds_base.labels)
train_loader_sampler = DataLoader(train_ds_base, 16, sampler=sampler, num_workers=0)

all_results = []


# ============================================================
# EXPERIMENT r3_01: TTA on best R2 model (r2_01) — no training
# ============================================================
print(f"\n{'=' * 70}")
print("r3_01: TTA on best R2 model (r2_01_effb0_smoothing_weighted_sampler)")
print(f"{'=' * 70}")

r2_best_ckpt = "models/best_r2_01_effb0_smoothing_weighted_sampler.pth"
model_tta = build_effb0_wide(dropout=0.3, pretrained=False).to(device)
model_tta.load_state_dict(torch.load(r2_best_ckpt, map_location=device))

# Baseline (no TTA)
val_acc_base, _, _, _ = evaluate(model_tta, val_loader)
test_acc_base, report_base, _, _ = evaluate(model_tta, test_loader)
print(f"  Baseline (no TTA): val={val_acc_base:.4f}  test={test_acc_base:.4f}")

# TTA with 8 augmented views
val_acc_tta, _ = evaluate_tta(model_tta, val_ds, n_aug=8)
test_acc_tta, report_tta = evaluate_tta(model_tta, test_ds, n_aug=8)
print(f"  TTA (8 views):     val={val_acc_tta:.4f}  test={test_acc_tta:.4f}")
print(report_tta)

all_results.append(
    {
        "name": "r3_01_tta_r2_best",
        "config": "TTA(8 views) on r2_01_effb0_smoothing_weighted_sampler",
        "val_acc": val_acc_tta,
        "test_acc": test_acc_tta,
        "report": report_tta,
        "checkpoint": r2_best_ckpt,
        "trained": False,
    }
)


# ============================================================
# EXPERIMENT r3_02: Ensemble of top-3 models
# ============================================================
print(f"\n{'=' * 70}")
print("r3_02: Ensemble of top-3 models (r2_01, exp5, exp2)")
print(f"{'=' * 70}")


def ensemble_predict(models_list, loader):
    for m in models_list:
        m.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            prob_sum = None
            for m in models_list:
                prob = F.softmax(m(imgs), dim=1)
                prob_sum = prob if prob_sum is None else prob_sum + prob
            all_probs.extend(torch.argmax(prob_sum, 1).cpu().numpy())
            all_true.extend(labels.numpy())
    acc = accuracy_score(all_true, all_probs)
    report = classification_report(
        all_true, all_probs, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    return acc, report


# Load top 3: r2_01 (effb0 wide head, dropout=0.3), exp5 (effb0 simple head, dropout=0.3), exp2 (resnet18)
model_r2_01 = build_effb0_wide(dropout=0.3, pretrained=False).to(device)
model_r2_01.load_state_dict(
    torch.load(
        "models/best_r2_01_effb0_smoothing_weighted_sampler.pth", map_location=device
    )
)

# exp5 used simple head: Dropout(0.3) -> Linear(1280, 5)
model_exp5 = build_effb0(dropout=0.3, pretrained=False).to(device)
model_exp5.load_state_dict(
    torch.load("models/best_exp5_efficientnetb0_smoothing.pth", map_location=device)
)

model_exp2 = build_resnet18(dropout=0.4, pretrained=False).to(device)
model_exp2.load_state_dict(
    torch.load("models/best_exp2_resnet18_label_smoothing.pth", map_location=device)
)

val_acc_ens, _ = ensemble_predict([model_r2_01, model_exp5, model_exp2], val_loader)
test_acc_ens, report_ens = ensemble_predict(
    [model_r2_01, model_exp5, model_exp2], test_loader
)
print(f"  Ensemble(r2_01+exp5+exp2): val={val_acc_ens:.4f}  test={test_acc_ens:.4f}")
print(report_ens)

all_results.append(
    {
        "name": "r3_02_ensemble_top3",
        "config": "Ensemble(r2_01_effb0+exp5_effb0+exp2_resnet18) avg softmax",
        "val_acc": val_acc_ens,
        "test_acc": test_acc_ens,
        "report": report_ens,
        "checkpoint": "ensemble",
        "trained": False,
    }
)

# Also try ensemble of just r2_01 + r2_06 (SGD nesterov)
model_r2_06 = build_effb0_wide(dropout=0.3, pretrained=False).to(device)
model_r2_06.load_state_dict(
    torch.load("models/best_r2_06_effb0_sgd_nesterov.pth", map_location=device)
)

val_acc_ens2, _ = ensemble_predict([model_r2_01, model_r2_06], val_loader)
test_acc_ens2, report_ens2 = ensemble_predict([model_r2_01, model_r2_06], test_loader)
print(f"  Ensemble(r2_01+r2_06): val={val_acc_ens2:.4f}  test={test_acc_ens2:.4f}")
print(report_ens2)
all_results.append(
    {
        "name": "r3_02b_ensemble_r2_01_r2_06",
        "config": "Ensemble(r2_01_effb0_adamw + r2_06_effb0_sgd) avg softmax",
        "val_acc": val_acc_ens2,
        "test_acc": test_acc_ens2,
        "report": report_ens2,
        "checkpoint": "ensemble",
        "trained": False,
    }
)


# ============================================================
# EXPERIMENT r3_03: EffB0 + WeightedSampler + lr=5e-4
# ============================================================
print(f"\n{'=' * 70}")
print("r3_03: EffB0 + WeightedSampler + lr=5e-4 (tune lr)")
print(f"{'=' * 70}")

model_r3_03 = build_effb0(dropout=0.3).to(device)
crit_r3 = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
opt_r3_03 = optim.AdamW(model_r3_03.parameters(), lr=5e-4, weight_decay=1e-4)
sch_r3_03 = optim.lr_scheduler.OneCycleLR(
    opt_r3_03,
    max_lr=5e-4,
    steps_per_epoch=len(train_loader_sampler),
    epochs=80,
    pct_start=0.1,
)
val_acc_r3_03, path_r3_03, elapsed_r3_03 = train_model(
    model_r3_03,
    train_loader_sampler,
    val_loader,
    crit_r3,
    opt_r3_03,
    sch_r3_03,
    80,
    20,
    "r3_03_effb0_lr5e4_sampler",
    True,
)
model_r3_03.load_state_dict(torch.load(path_r3_03, map_location=device))
test_acc_r3_03, report_r3_03, _, _ = evaluate(model_r3_03, test_loader)
print(
    f"  val_acc={val_acc_r3_03:.4f}  test_acc={test_acc_r3_03:.4f}  time={elapsed_r3_03 / 60:.1f}m"
)
print(report_r3_03)
all_results.append(
    {
        "name": "r3_03_effb0_lr5e4_sampler",
        "config": "EffB0(simple head) + AdamW lr=5e-4 + OneCycle + smoothing=0.1 + WeightedSampler",
        "val_acc": val_acc_r3_03,
        "test_acc": test_acc_r3_03,
        "report": report_r3_03,
        "checkpoint": path_r3_03,
        "trained": True,
        "elapsed_min": elapsed_r3_03 / 60,
    }
)


# ============================================================
# EXPERIMENT r3_04: EffB1 + WeightedSampler + lr=3e-4
# ============================================================
print(f"\n{'=' * 70}")
print("r3_04: EffB1 + WeightedSampler (slightly larger backbone)")
print(f"{'=' * 70}")

model_r3_04 = build_effb1(dropout=0.3).to(device)
opt_r3_04 = optim.AdamW(model_r3_04.parameters(), lr=3e-4, weight_decay=1e-4)
sch_r3_04 = optim.lr_scheduler.OneCycleLR(
    opt_r3_04,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader_sampler),
    epochs=80,
    pct_start=0.1,
)
val_acc_r3_04, path_r3_04, elapsed_r3_04 = train_model(
    model_r3_04,
    train_loader_sampler,
    val_loader,
    crit_r3,
    opt_r3_04,
    sch_r3_04,
    80,
    20,
    "r3_04_effb1_sampler",
    True,
)
model_r3_04.load_state_dict(torch.load(path_r3_04, map_location=device))
test_acc_r3_04, report_r3_04, _, _ = evaluate(model_r3_04, test_loader)
print(
    f"  val_acc={val_acc_r3_04:.4f}  test_acc={test_acc_r3_04:.4f}  time={elapsed_r3_04 / 60:.1f}m"
)
print(report_r3_04)
all_results.append(
    {
        "name": "r3_04_effb1_sampler",
        "config": "EffB1 + AdamW lr=3e-4 + OneCycle + smoothing=0.1 + WeightedSampler",
        "val_acc": val_acc_r3_04,
        "test_acc": test_acc_r3_04,
        "report": report_r3_04,
        "checkpoint": path_r3_04,
        "trained": True,
        "elapsed_min": elapsed_r3_04 / 60,
    }
)


# ============================================================
# EXPERIMENT r3_05: EffB0 + Mixup + WeightedSampler
# ============================================================
print(f"\n{'=' * 70}")
print("r3_05: EffB0 + Mixup(alpha=0.2) + WeightedSampler")
print(f"{'=' * 70}")

model_r3_05 = build_effb0(dropout=0.3).to(device)
opt_r3_05 = optim.AdamW(model_r3_05.parameters(), lr=3e-4, weight_decay=1e-4)
sch_r3_05 = optim.lr_scheduler.OneCycleLR(
    opt_r3_05,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader_sampler),
    epochs=80,
    pct_start=0.1,
)
val_acc_r3_05, path_r3_05, elapsed_r3_05 = train_model(
    model_r3_05,
    train_loader_sampler,
    val_loader,
    crit_r3,
    opt_r3_05,
    sch_r3_05,
    80,
    20,
    "r3_05_effb0_mixup_sampler",
    True,
    use_mixup=True,
)
model_r3_05.load_state_dict(torch.load(path_r3_05, map_location=device))
test_acc_r3_05, report_r3_05, _, _ = evaluate(model_r3_05, test_loader)
print(
    f"  val_acc={val_acc_r3_05:.4f}  test_acc={test_acc_r3_05:.4f}  time={elapsed_r3_05 / 60:.1f}m"
)
print(report_r3_05)
all_results.append(
    {
        "name": "r3_05_effb0_mixup_sampler",
        "config": "EffB0 + Mixup(alpha=0.2) + AdamW lr=3e-4 + OneCycle + smoothing=0.1 + WeightedSampler",
        "val_acc": val_acc_r3_05,
        "test_acc": test_acc_r3_05,
        "report": report_r3_05,
        "checkpoint": path_r3_05,
        "trained": True,
        "elapsed_min": elapsed_r3_05 / 60,
    }
)


# ============================================================
# EXPERIMENT r3_06: EffB0 + Focal Loss + WeightedSampler
# ============================================================
print(f"\n{'=' * 70}")
print("r3_06: EffB0 + FocalLoss(gamma=2) + WeightedSampler")
print(f"{'=' * 70}")

model_r3_06 = build_effb0(dropout=0.3).to(device)
focal_crit = FocalLoss(weight=cw, gamma=2.0, label_smoothing=0.1)
opt_r3_06 = optim.AdamW(model_r3_06.parameters(), lr=3e-4, weight_decay=1e-4)
sch_r3_06 = optim.lr_scheduler.OneCycleLR(
    opt_r3_06,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader_sampler),
    epochs=80,
    pct_start=0.1,
)
val_acc_r3_06, path_r3_06, elapsed_r3_06 = train_model(
    model_r3_06,
    train_loader_sampler,
    val_loader,
    focal_crit,
    opt_r3_06,
    sch_r3_06,
    80,
    20,
    "r3_06_effb0_focal_sampler",
    True,
)
model_r3_06.load_state_dict(torch.load(path_r3_06, map_location=device))
test_acc_r3_06, report_r3_06, _, _ = evaluate(model_r3_06, test_loader)
print(
    f"  val_acc={val_acc_r3_06:.4f}  test_acc={test_acc_r3_06:.4f}  time={elapsed_r3_06 / 60:.1f}m"
)
print(report_r3_06)
all_results.append(
    {
        "name": "r3_06_effb0_focal_sampler",
        "config": "EffB0 + FocalLoss(gamma=2, smoothing=0.1) + AdamW lr=3e-4 + OneCycle + WeightedSampler",
        "val_acc": val_acc_r3_06,
        "test_acc": test_acc_r3_06,
        "report": report_r3_06,
        "checkpoint": path_r3_06,
        "trained": True,
        "elapsed_min": elapsed_r3_06 / 60,
    }
)


# ============================================================
# Summary
# ============================================================
all_results.sort(key=lambda x: x["test_acc"], reverse=True)
print(f"\n{'=' * 70}")
print("ROUND 3 RANKINGS BY TEST ACCURACY")
print(f"{'=' * 70}")
print(f"{'Rank':>4} {'Name':>50} {'Val Acc':>9} {'Test Acc':>9}")
print("-" * 75)
for i, r in enumerate(all_results):
    print(f"{i + 1:>4} {r['name']:>50} {r['val_acc']:>9.4f} {r['test_acc']:>9.4f}")

print(f"\nBest R3: {all_results[0]['name']}  test_acc={all_results[0]['test_acc']:.4f}")
print("Prev best (R2): r2_01_effb0_smoothing_weighted_sampler  test_acc=0.5276")

import json

with open("r3_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved to r3_results.json")

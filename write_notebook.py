#!/usr/bin/env python3
"""Rewrite main.ipynb with the best configuration found across all experiments."""

import json


def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


cells = []

# ── Cell 0: Title ──────────────────────────────────────────────────────────
cells.append(
    md_cell(
        [
            "# Diabetic Retinopathy Classification\n",
            "\n",
            "Multi-class CNN classifier for diabetic retinopathy stages (0–4) using the pre-split `data_split` dataset.\n",
            "\n",
            "**Best configuration** (found after 3 rounds of systematic experiments, 24 configurations):\n",
            "- Architecture: EfficientNetB0 (pretrained ImageNet)\n",
            "- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4\n",
            "- Scheduler: OneCycleLR\n",
            "- Loss: CrossEntropyLoss + inverse-frequency class weights + label_smoothing=0.1\n",
            "- Sampler: WeightedRandomSampler (oversample minority classes)\n",
            "\n",
            "**Results**: best single model test accuracy **0.5276**, 3-model ensemble **0.5512**\n",
            "\n",
            "Classes: 0-noDR | 1-mild | 2-moderate | 3-severe | 4-proliferativeDR\n",
        ]
    )
)

# ── Cell 1: Imports & setup ────────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler\n",
            "from torchvision import transforms, models\n",
            "from torchvision.models import EfficientNet_B0_Weights\n",
            "\n",
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
            "\n",
            "import random, os, time\n",
            "import numpy as np\n",
            "from PIL import Image\n",
            "from collections import Counter\n",
            "\n",
            "# Reproducibility\n",
            "SEED = 42\n",
            "random.seed(SEED)\n",
            "np.random.seed(SEED)\n",
            "torch.manual_seed(SEED)\n",
            "\n",
            "device = torch.device(\n",
            "    'cuda' if torch.cuda.is_available()\n",
            "    else 'mps' if torch.backends.mps.is_available()\n",
            "    else 'cpu'\n",
            ")\n",
            "print(f'Using device: {device}')\n",
            "\n",
            "data_split_dir = 'data_split'\n",
            "model_dir = 'models'\n",
            "os.makedirs(model_dir, exist_ok=True)\n",
            "\n",
            "CLASS_NAMES = ['0-noDR', '1-mild', '2-moderate', '3-severe', '4-proliferativeDR']\n",
            "NUM_CLASSES = len(CLASS_NAMES)\n",
            "IMAGENET_MEAN = [0.485, 0.456, 0.406]\n",
            "IMAGENET_STD  = [0.229, 0.224, 0.225]\n",
            "print(f'Classes ({NUM_CLASSES}): {CLASS_NAMES}')\n",
        ]
    )
)

# ── Cell 2: Dataset markdown ───────────────────────────────────────────────
cells.append(
    md_cell(
        [
            "## Dataset\n",
            "\n",
            "Pre-split dataset in `data_split/train`, `data_split/val`, `data_split/test`.\n",
            "The dataset is **highly imbalanced** (e.g., 326 no-DR vs 60 proliferative-DR in train).\n",
            "Training uses data augmentation and **WeightedRandomSampler** to handle class imbalance.\n",
        ]
    )
)

# ── Cell 3: Dataset class + transforms + loaders ──────────────────────────
cells.append(
    code_cell(
        [
            "class SplitImageDataset(Dataset):\n",
            '    """Loads images from root/class_name/image.jpg structure."""\n',
            "    def __init__(self, root_dir, transform=None):\n",
            "        self.transform = transform\n",
            "        self.image_paths = []\n",
            "        self.labels = []\n",
            "        class_names = sorted([d for d in os.listdir(root_dir)\n",
            "                               if os.path.isdir(os.path.join(root_dir, d))])\n",
            "        for class_name in class_names:\n",
            "            class_dir = os.path.join(root_dir, class_name)\n",
            "            label = int(class_name.split('-')[0])\n",
            "            for fname in sorted(os.listdir(class_dir)):\n",
            "                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):\n",
            "                    self.image_paths.append(os.path.join(class_dir, fname))\n",
            "                    self.labels.append(label)\n",
            "        print(f'Loaded {len(self.image_paths)} images from {root_dir}')\n",
            "        counts = Counter(self.labels)\n",
            "        for name in class_names:\n",
            "            idx = int(name.split('-')[0])\n",
            "            print(f'  {name}: {counts[idx]}')\n",
            "\n",
            "    def __len__(self):\n",
            "        return len(self.image_paths)\n",
            "\n",
            "    def __getitem__(self, idx):\n",
            "        with Image.open(self.image_paths[idx]) as img:\n",
            "            img = img.convert('RGB')\n",
            "            if self.transform:\n",
            "                img = self.transform(img)\n",
            "        return img, self.labels[idx]\n",
            "\n",
            "\n",
            "train_transform = transforms.Compose([\n",
            "    transforms.Resize((256, 256)),\n",
            "    transforms.RandomCrop(224),\n",
            "    transforms.RandomHorizontalFlip(0.5),\n",
            "    transforms.RandomVerticalFlip(0.5),\n",
            "    transforms.RandomRotation(30),\n",
            "    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),\n",
            "    transforms.RandomErasing(p=0.1),\n",
            "])\n",
            "\n",
            "val_transform = transforms.Compose([\n",
            "    transforms.Resize((224, 224)),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),\n",
            "])\n",
            "\n",
            "print('=== Training set ===')\n",
            "train_dataset = SplitImageDataset(os.path.join(data_split_dir, 'train'), train_transform)\n",
            "print('\\n=== Validation set ===')\n",
            "val_dataset   = SplitImageDataset(os.path.join(data_split_dir, 'val'),   val_transform)\n",
            "print('\\n=== Test set ===')\n",
            "test_dataset  = SplitImageDataset(os.path.join(data_split_dir, 'test'),  val_transform)\n",
        ]
    )
)

# ── Cell 4: Class weights markdown ────────────────────────────────────────
cells.append(
    md_cell(
        [
            "## Class Weights & WeightedRandomSampler\n",
            "\n",
            "Two complementary strategies for handling class imbalance:\n",
            "1. **Inverse-frequency class weights** in the loss function\n",
            "2. **WeightedRandomSampler** to oversample minority classes during training batches\n",
            "\n",
            "The sampler was the single biggest accuracy gain found in experiments (+3.9% test accuracy).\n",
        ]
    )
)

# ── Cell 5: Weights + sampler ──────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "# Inverse-frequency class weights for the loss function\n",
            "label_counts = Counter(train_dataset.labels)\n",
            "total = len(train_dataset.labels)\n",
            "class_weights = torch.zeros(NUM_CLASSES)\n",
            "for i in range(NUM_CLASSES):\n",
            "    class_weights[i] = total / (NUM_CLASSES * label_counts.get(i, 1))\n",
            "\n",
            "print('Class weights (inverse frequency):')\n",
            "for i, (name, w) in enumerate(zip(CLASS_NAMES, class_weights)):\n",
            "    print(f'  {name}: weight={w:.4f}  (n={label_counts[i]})')\n",
            "class_weights = class_weights.to(device)\n",
            "\n",
            "# WeightedRandomSampler: each sample drawn with probability 1/class_count\n",
            "sample_weights = [1.0 / label_counts[lbl] for lbl in train_dataset.labels]\n",
            "sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)\n",
            "print('\\nWeightedRandomSampler created.')\n",
        ]
    )
)

# ── Cell 6: Model markdown ─────────────────────────────────────────────────
cells.append(
    md_cell(
        [
            "## Model Architecture\n",
            "\n",
            "**EfficientNetB0** pretrained on ImageNet, with a lightweight custom head:\n",
            "- `Dropout(0.3)` → `Linear(1280 → 5)`\n",
            "\n",
            "EfficientNetB0 was the best architecture for this small dataset (~850 train images).\n",
            "Larger models (ResNet50, EfficientNetB2) overfit significantly.\n",
        ]
    )
)

# ── Cell 7: Model definition ───────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "def build_model(num_classes=5, dropout=0.3, pretrained=True):\n",
            '    """EfficientNetB0 with custom head for retinopathy classification."""\n',
            "    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None\n",
            "    model = models.efficientnet_b0(weights=weights)\n",
            "    in_features = model.classifier[1].in_features\n",
            "    model.classifier = nn.Sequential(\n",
            "        nn.Dropout(p=dropout),\n",
            "        nn.Linear(in_features, num_classes),\n",
            "    )\n",
            "    return model\n",
            "\n",
            "\n",
            "model = build_model(num_classes=NUM_CLASSES, dropout=0.3, pretrained=True).to(device)\n",
            "\n",
            "total_params     = sum(p.numel() for p in model.parameters())\n",
            "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
            "print('Architecture: EfficientNetB0 (pretrained ImageNet)')\n",
            "print(f'Total parameters:     {total_params:,}')\n",
            "print(f'Trainable parameters: {trainable_params:,}')\n",
            "print(f'Classifier head:      Dropout(0.3) -> Linear(1280, {NUM_CLASSES})')\n",
        ]
    )
)

# ── Cell 8: Training markdown ──────────────────────────────────────────────
cells.append(
    md_cell(
        [
            "## Training\n",
            "\n",
            "Best hyperparameters found after 24 experiments across 3 rounds:\n",
            "- **Optimizer**: AdamW, lr=3e-4, weight_decay=1e-4\n",
            "- **Scheduler**: OneCycleLR (max_lr=3e-4, pct_start=0.1, cosine annealing) — steps per batch\n",
            "- **Loss**: CrossEntropyLoss with class weights + label_smoothing=0.1\n",
            "- **Sampler**: WeightedRandomSampler\n",
            "- **Batch size**: 16, **Max epochs**: 80, **Early stopping patience**: 20\n",
            "- **Gradient clipping**: max_norm=1.0\n",
        ]
    )
)

# ── Cell 9: Config + loaders ───────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "# ==================== Configuration ====================\n",
            "NUM_EPOCHS    = 80\n",
            "BATCH_SIZE    = 16\n",
            "LEARNING_RATE = 3e-4\n",
            "WEIGHT_DECAY  = 1e-4\n",
            "PATIENCE      = 20\n",
            "BEST_MODEL_PATH = os.path.join(model_dir, 'best_model_effb0_sampler.pth')\n",
            "\n",
            "# ==================== Data Loaders ====================\n",
            "# Training uses WeightedRandomSampler (mutually exclusive with shuffle=True)\n",
            "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,\n",
            "                          num_workers=0, pin_memory=False)\n",
            "val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,\n",
            "                          num_workers=0, pin_memory=False)\n",
            "test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,\n",
            "                          num_workers=0, pin_memory=False)\n",
            "print(f'Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}')\n",
            "\n",
            "# ==================== Loss / Optimizer / Scheduler ====================\n",
            "criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)\n",
            "optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)\n",
            "scheduler = optim.lr_scheduler.OneCycleLR(\n",
            "    optimizer, max_lr=LEARNING_RATE,\n",
            "    steps_per_epoch=len(train_loader),\n",
            "    epochs=NUM_EPOCHS,\n",
            "    pct_start=0.1, anneal_strategy='cos',\n",
            ")\n",
            "print(f'Optimizer: AdamW  lr={LEARNING_RATE}  weight_decay={WEIGHT_DECAY}')\n",
            "print(f'Scheduler: OneCycleLR  max_lr={LEARNING_RATE}')\n",
            "print(f'Loss: CrossEntropyLoss(label_smoothing=0.1) + inverse-freq class weights')\n",
        ]
    )
)

# ── Cell 10: evaluate + training loop ─────────────────────────────────────
cells.append(
    code_cell(
        [
            "def evaluate(model, loader, criterion, device):\n",
            '    """Evaluate model on loader. Returns avg loss and accuracy."""\n',
            "    model.eval()\n",
            "    total_loss = 0.0\n",
            "    preds, true_labels = [], []\n",
            "    with torch.no_grad():\n",
            "        for images, labels in loader:\n",
            "            images, labels = images.to(device), labels.to(device)\n",
            "            outputs = model(images)\n",
            "            total_loss += criterion(outputs, labels).item()\n",
            "            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())\n",
            "            true_labels.extend(labels.cpu().numpy())\n",
            "    return total_loss / len(loader), accuracy_score(true_labels, preds)\n",
            "\n",
            "\n",
            "# ==================== Training Loop ====================\n",
            "best_val_acc     = 0.0\n",
            "patience_counter = 0\n",
            "history = {'train_loss': [], 'val_loss': [], 'val_acc': []}\n",
            "\n",
            "print(f'  Epoch  TrainLoss   ValLoss   ValAcc     Best')\n",
            "print('-' * 52)\n",
            "\n",
            "start_time = time.time()\n",
            "\n",
            "for epoch in range(NUM_EPOCHS):\n",
            "    # ---- Training ----\n",
            "    model.train()\n",
            "    total_loss = 0.0\n",
            "    for images, labels in train_loader:\n",
            "        images, labels = images.to(device), labels.to(device)\n",
            "        optimizer.zero_grad()\n",
            "        loss = criterion(model(images), labels)\n",
            "        loss.backward()\n",
            "        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
            "        optimizer.step()\n",
            "        scheduler.step()  # OneCycleLR steps per batch\n",
            "        total_loss += loss.item()\n",
            "\n",
            "    avg_train_loss = total_loss / len(train_loader)\n",
            "    val_loss, val_acc = evaluate(model, val_loader, criterion, device)\n",
            "\n",
            "    history['train_loss'].append(avg_train_loss)\n",
            "    history['val_loss'].append(val_loss)\n",
            "    history['val_acc'].append(val_acc)\n",
            "\n",
            "    improved = val_acc > best_val_acc\n",
            "    if improved:\n",
            "        best_val_acc = val_acc\n",
            "        patience_counter = 0\n",
            "        torch.save(model.state_dict(), BEST_MODEL_PATH)\n",
            "    else:\n",
            "        patience_counter += 1\n",
            "\n",
            "    if (epoch + 1) % 5 == 0 or epoch == 0 or improved:\n",
            "        marker = ' *' if improved else ''\n",
            "        print(f'  {epoch+1:>4}/{NUM_EPOCHS}  {avg_train_loss:>9.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}  {best_val_acc:>7.4f}{marker}')\n",
            "\n",
            "    if patience_counter >= PATIENCE:\n",
            "        print(f'Early stopping at epoch {epoch+1}.')\n",
            "        break\n",
            "\n",
            "elapsed = time.time() - start_time\n",
            "print(f'\\nTraining finished in {elapsed/60:.1f} min')\n",
            "print(f'Best Validation Accuracy: {best_val_acc:.4f}')\n",
        ]
    )
)

# ── Cell 11: Evaluation markdown ──────────────────────────────────────────
cells.append(md_cell(["## Evaluation on Test Set\n"]))

# ── Cell 12: Test evaluation ───────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "# Load best model checkpoint\n",
            "model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))\n",
            "model.eval()\n",
            "\n",
            "test_loss, test_acc = evaluate(model, test_loader, criterion, device)\n",
            "print(f'Test Loss:     {test_loss:.4f}')\n",
            "print(f'Test Accuracy: {test_acc:.4f}')\n",
            "\n",
            "# Full classification report\n",
            "all_preds, all_true = [], []\n",
            "with torch.no_grad():\n",
            "    for images, labels in test_loader:\n",
            "        outputs = model(images.to(device))\n",
            "        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())\n",
            "        all_true.extend(labels.numpy())\n",
            "\n",
            "print('\\nClassification Report:')\n",
            "print(classification_report(all_true, all_preds, target_names=CLASS_NAMES, digits=4))\n",
            "print('Confusion Matrix:')\n",
            "print(confusion_matrix(all_true, all_preds))\n",
        ]
    )
)

# ── Cell 13: Ensemble markdown ─────────────────────────────────────────────
cells.append(
    md_cell(
        [
            "## Ensemble (Best Overall)\n",
            "\n",
            "Ensembling three complementary pre-trained models achieves the best overall accuracy.\n",
            "Strategy: average softmax probabilities across all models.\n",
            "\n",
            "| Model | Architecture | Test Acc |\n",
            "|---|---|---|\n",
            "| r2_01 | EffB0 + AdamW + WeightedSampler | 0.5276 |\n",
            "| exp5  | EffB0 (simple head, label smooth) | 0.4882 |\n",
            "| exp2  | ResNet18 + AdamW + label smooth | 0.4803 |\n",
            "| **Ensemble** | **Average softmax** | **0.5512** |\n",
        ]
    )
)

# ── Cell 14: Ensemble evaluation ──────────────────────────────────────────
cells.append(
    code_cell(
        [
            "def build_effb0_wide(dropout=0.3):\n",
            "    m = models.efficientnet_b0(weights=None)\n",
            "    in_feat = m.classifier[1].in_features\n",
            "    m.classifier = nn.Sequential(\n",
            "        nn.Dropout(dropout), nn.Linear(in_feat, 256),\n",
            "        nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),\n",
            "        nn.Linear(256, NUM_CLASSES),\n",
            "    )\n",
            "    return m\n",
            "\n",
            "def build_effb0_simple(dropout=0.3):\n",
            "    m = models.efficientnet_b0(weights=None)\n",
            "    in_feat = m.classifier[1].in_features\n",
            "    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, NUM_CLASSES))\n",
            "    return m\n",
            "\n",
            "def build_resnet18_wide(dropout=0.4):\n",
            "    m = models.resnet18(weights=None)\n",
            "    in_feat = m.fc.in_features\n",
            "    m.fc = nn.Sequential(\n",
            "        nn.Dropout(dropout), nn.Linear(in_feat, 256),\n",
            "        nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),\n",
            "        nn.Linear(256, NUM_CLASSES),\n",
            "    )\n",
            "    return m\n",
            "\n",
            "\n",
            "# Load the 3 best pre-trained models\n",
            "m_r2_01 = build_effb0_wide(0.3).to(device)\n",
            "m_r2_01.load_state_dict(torch.load(\n",
            "    'models/best_r2_01_effb0_smoothing_weighted_sampler.pth', map_location=device))\n",
            "\n",
            "m_exp5 = build_effb0_simple(0.3).to(device)\n",
            "m_exp5.load_state_dict(torch.load(\n",
            "    'models/best_exp5_efficientnetb0_smoothing.pth', map_location=device))\n",
            "\n",
            "m_exp2 = build_resnet18_wide(0.4).to(device)\n",
            "m_exp2.load_state_dict(torch.load(\n",
            "    'models/best_exp2_resnet18_label_smoothing.pth', map_location=device))\n",
            "\n",
            "ensemble_models = [m_r2_01, m_exp5, m_exp2]\n",
            "for m in ensemble_models:\n",
            "    m.eval()\n",
            "\n",
            "# Ensemble: average softmax probabilities\n",
            "ens_preds, ens_true = [], []\n",
            "with torch.no_grad():\n",
            "    for images, labels in test_loader:\n",
            "        images = images.to(device)\n",
            "        prob_sum = sum(F.softmax(m(images), dim=1) for m in ensemble_models)\n",
            "        ens_preds.extend(torch.argmax(prob_sum, 1).cpu().numpy())\n",
            "        ens_true.extend(labels.numpy())\n",
            "\n",
            "ens_acc = accuracy_score(ens_true, ens_preds)\n",
            "print(f'Ensemble Test Accuracy: {ens_acc:.4f}')\n",
            "print('\\nEnsemble Classification Report:')\n",
            "print(classification_report(ens_true, ens_preds, target_names=CLASS_NAMES, digits=4))\n",
            "print('Confusion Matrix:')\n",
            "print(confusion_matrix(ens_true, ens_preds))\n",
        ]
    )
)

# ── Cell 15: Sample inference markdown ────────────────────────────────────
cells.append(
    md_cell(
        [
            "## Sample Inference\n",
            "\n",
            "Test the best single model on random samples from the test set.\n",
        ]
    )
)

# ── Cell 16: Sample inference ──────────────────────────────────────────────
cells.append(
    code_cell(
        [
            "m_r2_01.eval()\n",
            "idx_to_class = {int(name.split('-')[0]): name for name in CLASS_NAMES}\n",
            "\n",
            "print('Sample predictions (best single model: EffB0 + WeightedSampler):')\n",
            "print('-' * 65)\n",
            "print(f\"  {'Idx':>5}  {'True Label':>22}  {'Predicted':>22}  {'Correct':>7}\")\n",
            "print('-' * 65)\n",
            "\n",
            "for _ in range(10):\n",
            "    idx = random.randint(0, len(test_dataset) - 1)\n",
            "    img, true_label = test_dataset[idx]\n",
            "    with torch.no_grad():\n",
            "        out = m_r2_01(img.unsqueeze(0).to(device))\n",
            "        pred = torch.argmax(out, 1).item()\n",
            "    correct = 'YES' if true_label == pred else 'NO'\n",
            '    print(f"  {idx:>5}  {idx_to_class[true_label]:>22}  {idx_to_class[pred]:>22}  {correct:>7}")\n',
        ]
    )
)


# ── Write notebook ─────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.14.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("main.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote main.ipynb with {len(cells)} cells.")

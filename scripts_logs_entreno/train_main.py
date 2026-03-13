"""
Train the best configuration (EfficientNetB0 + WeightedSampler + OneCycleLR)
and save the model. This script produces the outputs for main.ipynb.
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

data_split_dir = "data_split"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

CLASS_NAMES = ["0-noDR", "1-mild", "2-moderate", "3-severe", "4-proliferativeDR"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")


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
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
        counts = Counter(self.labels)
        for name in class_names:
            idx = int(name.split("-")[0])
            print(f"  {name}: {counts[idx]}")

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

print("=== Training set ===")
train_dataset = SplitImageDataset(
    os.path.join(data_split_dir, "train"), train_transform
)
print("\n=== Validation set ===")
val_dataset = SplitImageDataset(os.path.join(data_split_dir, "val"), val_transform)
print("\n=== Test set ===")
test_dataset = SplitImageDataset(os.path.join(data_split_dir, "test"), val_transform)

# Class weights
label_counts = Counter(train_dataset.labels)
total = len(train_dataset.labels)
class_weights = torch.zeros(NUM_CLASSES)
for i in range(NUM_CLASSES):
    class_weights[i] = total / (NUM_CLASSES * label_counts.get(i, 1))

print("\nClass weights (inverse frequency):")
for i, (name, w) in enumerate(zip(CLASS_NAMES, class_weights)):
    print(f"  {name}: weight={w:.4f}  (n={label_counts[i]})")
class_weights = class_weights.to(device)

sample_weights = [1.0 / label_counts[lbl] for lbl in train_dataset.labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
print("\nWeightedRandomSampler created.")


def build_model(num_classes=5, dropout=0.3, pretrained=True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


model = build_model(num_classes=NUM_CLASSES, dropout=0.3, pretrained=True).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Architecture: EfficientNetB0 (pretrained ImageNet)")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Config
NUM_EPOCHS = 40
BATCH_SIZE = 32  # larger batch for speed with more data
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
BEST_MODEL_PATH = os.path.join(model_dir, "best_model_effb0_sampler.pth")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=False,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
)
print(
    f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.1,
    anneal_strategy="cos",
)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, true_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(true_labels, preds)


best_val_acc = 0.0
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_acc": []}

print(f"  Epoch  TrainLoss   ValLoss   ValAcc     Best")
print("-" * 52)

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
        marker = " *" if improved else ""
        print(
            f"  {epoch + 1:>4}/{NUM_EPOCHS}  {avg_train_loss:>9.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}  {best_val_acc:>7.4f}{marker}"
        )

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}.")
        break

elapsed = time.time() - start_time
print(f"\nTraining finished in {elapsed / 60:.1f} min")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# Evaluate on test set
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

all_preds, all_true = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_true.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=CLASS_NAMES, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_true, all_preds))

# Save results
import json

results = {
    "test_accuracy": test_acc,
    "best_val_accuracy": best_val_acc,
    "epochs_trained": len(history["val_acc"]),
    "history": history,
}
json.dump(results, open("train_main_results.json", "w"), indent=2)
print("\nResults saved to train_main_results.json")

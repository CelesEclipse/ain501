import os
import cv2
import json
import numpy as np
from pathlib import Path
from parse import model_to_dict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

"""
Use Albumentations method
This is optional, use only when the dataset is small
as a data augmentation method
"""
from typing import Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_list: list[Any] = [
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
]

val_list: list[Any] = [
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
]

train_transform = A.Compose(train_list)
val_transform = A.Compose(val_list)

IMG_DIR = "data/images"
MASK_DIR = "data/masks"
JSON_DIR = "parsed_model.json"

"""
Dataset class
"""

class DocumentDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.images = sorted([
            f.name for f in self.img_dir.iterdir()
            if f.suffix in [".jpg", ".png", ".tif"]
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        mask_name = Path(img_name).with_suffix(".png").name
        img = cv2.imread(str(self.img_dir / img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / mask_name), cv2.IMREAD_GRAYSCALE)

        # Binarize mask
        mask = (mask > 127).astype(np.float32)

        # Apply transform
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return img, mask

"""
Split 80/10/10
"""
full_dataset = DocumentDataset(
    img_dir  = IMG_DIR,
    mask_dir = MASK_DIR,
    transform= train_transform
)

total = len(full_dataset)
train_size = int(0.8 * total)
value_size = int(0.1 * total)
test_size = total - train_size - value_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, value_size, test_size])

val_ds.dataset.transform  = val_transform
test_ds.dataset.transform = val_transform
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

"""
Model
"""
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)

"""
Instantiate model and print
"""
device = torch.device("cpu")
model = model.to(device)
model_json = model_to_dict(model)
with open(JSON_DIR, "w", encoding="utf-8") as f:
    json.dump(model_json, f, indent=2)

# freeze encoders - phase 1
for param in model.encoder.parameters():
    param.requires_grad = False

# total_params = sum(p.numel() for p in model.parameters())
# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# frozen_params = total_params - trainable
# print(f"Total:     {total_params:,}")
# print(f"Trainable: {trainable:,}")
# print(f"Frozen:    {frozen_params:,}")  # make sure you're printing frozen_params not something else

# # unfreeze
def unfreeze_encoder(model, lr_encoder=1e-5, lr_decoder=1e-4):
    # Unfreeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = True

    # Separate param groups with different learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr_encoder},
        {"params": model.decoder.parameters(), "lr": lr_decoder},
        {"params": model.segmentation_head.parameters(), "lr": lr_decoder},
    ])
    return optimizer

# dummy_input = torch.randn(1, 3, 256, 256)   # batch=1, RGB, 256x256
# output      = model(dummy_input)

# print(f"Input  shape: {dummy_input.shape}")  # [1, 3, 256, 256]
# print(f"Output shape: {output.shape}")       # [1, 1, 256, 256] ← correct

"""
Loss functions: BCE + DICE
"""
import torch.nn as nn
bce_loss_fn = nn.BCEWithLogitsLoss()
dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

def combined_loss(pred, targets):
    bce = bce_loss_fn(pred, targets)
    dice = dice_loss_fn(pred, targets)
    return 0.5 * bce + 0.5 * dice

def compute_iou(predictions, targets, threshold=0.5):
    # Apply sigmoid to get probabilities, then threshold
    preds = (torch.sigmoid(predictions) > threshold).float()

    intersection = (preds * targets).sum()
    union        = (preds + targets).sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)  # 1e-6 avoids division by zero
    return iou.item()

def compute_dice(predictions, targets, threshold=0.5):
    preds = (torch.sigmoid(predictions) > threshold).float()

    intersection = (preds * targets).sum()
    dice = (2 * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)
    return dice.item()

# # Fake a batch to confirm loss and metrics work
# dummy_pred   = torch.randn(4, 1, 256, 256)   # model output
# dummy_target = torch.randint(0, 2, (4, 1, 256, 256)).float()  # binary mask

# loss = combined_loss(dummy_pred, dummy_target)
# iou  = compute_iou(dummy_pred, dummy_target)
# dice = compute_dice(dummy_pred, dummy_target)

# print(f"Loss: {loss:.4f} | IoU: {iou:.4f} | Dice: {dice:.4f}")

"""
Training loops
"""
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_iou, total_dice = 0, 0, 0

    for images, masks in tqdm(loader, desc="Train"):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss        = combined_loss(predictions, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou  += compute_iou(predictions, masks)
        total_dice += compute_dice(predictions, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def validate(model, loader, device):
    model.eval()
    total_loss, total_iou, total_dice = 0, 0, 0

    with torch.no_grad():                      # no gradients needed for val
        for images, masks in tqdm(loader, desc="Val  "):
            images = images.to(device)
            masks  = masks.to(device)

            predictions = model(images)
            loss        = combined_loss(predictions, masks)

            total_loss += loss.item()
            total_iou  += compute_iou(predictions, masks)
            total_dice += compute_dice(predictions, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def save_best_model(model, val_iou, best_iou, path="best_model.pth"):
    if val_iou > best_iou:
        torch.save(model.state_dict(), path)
        print(f"  ✓ Saved best model (IoU: {val_iou:.4f})")
        return val_iou                         # update best
    return best_iou                            # keep old best

# Optimizer for decoder only (encoder is frozen)
optimizer_phase1 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
scheduler_phase1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_phase1, mode="max", patience=3, factor=0.5
)
# mode="max" because we're tracking IoU (higher = better)

PHASE1_EPOCHS = 10
best_iou      = 0.0

print("=" * 50)
print("PHASE 1 — Decoder only")
print("=" * 50)

for epoch in range(PHASE1_EPOCHS):
    print(f"\nEpoch {epoch+1}/{PHASE1_EPOCHS}")

    train_loss, train_iou, train_dice = train_one_epoch(model, train_loader, optimizer_phase1, device)
    val_loss,   val_iou,   val_dice   = validate(model, val_loader, device)

    print(f"  Train → Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
    print(f"  Val   → Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

    scheduler_phase1.step(val_iou)
    best_iou = save_best_model(model, val_iou, best_iou)

"""
Phase 2
"""
# Unfreeze encoder with lower LR
for param in model.encoder.parameters():
    param.requires_grad = True

optimizer_phase2 = torch.optim.AdamW([
    {"params": model.encoder.parameters(),          "lr": 1e-5},  # gentle
    {"params": model.decoder.parameters(),          "lr": 1e-4},  # normal
    {"params": model.segmentation_head.parameters(),"lr": 1e-4},
])
scheduler_phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_phase2, T_max=20
)
# CosineAnnealingLR smoothly decays LR → good for fine-tuning

PHASE2_EPOCHS = 20

print("\n" + "=" * 50)
print("PHASE 2 — Full model")
print("=" * 50)

for epoch in range(PHASE2_EPOCHS):
    print(f"\nEpoch {epoch+1}/{PHASE2_EPOCHS}")

    train_loss, train_iou, train_dice = train_one_epoch(model, train_loader, optimizer_phase2, device)
    val_loss,   val_iou,   val_dice   = validate(model, val_loader, device)

    print(f"  Train → Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}")
    print(f"  Val   → Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

    scheduler_phase2.step()
    best_iou = save_best_model(model, val_iou, best_iou)

print("\nTraining complete. Best IoU:", round(best_iou, 4))

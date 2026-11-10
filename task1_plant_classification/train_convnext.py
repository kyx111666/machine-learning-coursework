"""Five-fold ImageNet-initialized ConvNeXt-Tiny training with eight-view TTA."""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import InterpolationMode

from data import (CLASSES, find_data, fold_ids, image_hash, write_submission)

classes = CLASSES.tolist()
SEED = 2026
IMG_SIZE = 320
BATCH_SIZE = 12
EPOCHS = 20
N_FOLDS = 5
PATIENCE = 5

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class SquarePad:
    """先把图片补成正方形，避免中心裁剪切掉叶片。"""

    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return transforms.functional.pad(image, [left, top, right, bottom], fill=0)


train_tf = transforms.Compose([
    SquarePad(),
    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.85, 1.0),
        ratio=(0.95, 1.05),
        interpolation=InterpolationMode.BICUBIC,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    transforms.ToTensor(),
    normalize,
])

valid_tf = transforms.Compose([
    SquarePad(),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])


class PlantDataset(Dataset):
    def __init__(self, files, labels=None, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        if self.labels is None:
            return image, self.files[index].name
        return image, int(self.labels[index])


def make_model():
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(classes))
    return model

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)

        # 反向传播计算梯度，然后更新网络权重
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * len(labels)

    return loss_sum / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    labels_all = []
    probs_all = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)

        loss_sum += loss.item() * len(labels)
        labels_all.append(labels.cpu().numpy())
        probs_all.append(logits.softmax(1).cpu().numpy())

    labels_all = np.concatenate(labels_all)
    probs_all = np.concatenate(probs_all)
    preds_all = probs_all.argmax(1)
    score = f1_score(labels_all, preds_all, average="macro")
    return loss_sum / len(loader.dataset), score, probs_all

@torch.no_grad()
def tta_probabilities(model, images, original_probs=None):
    # 植物方向不影响类别，平均 4 个旋转和它们的水平翻转。
    total = original_probs.clone() if original_probs is not None else model(images).softmax(1)
    total += model(torch.flip(images, dims=[3])).softmax(1)
    for k in (1, 2, 3):
        rotated = torch.rot90(images, k=k, dims=[2, 3])
        total += model(rotated).softmax(1)
        total += model(torch.flip(rotated, dims=[3])).softmax(1)
    return total / 8

@torch.no_grad()
def evaluate_with_tta(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    labels_all = []
    probs_all = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss_sum += loss_fn(logits, labels).item() * len(labels)
        probs = tta_probabilities(model, images, logits.softmax(1))
        labels_all.append(labels.cpu().numpy())
        probs_all.append(probs.cpu().numpy())

    labels_all = np.concatenate(labels_all)
    probs_all = np.concatenate(probs_all)
    score = f1_score(labels_all, probs_all.argmax(1), average="macro")
    return loss_sum / len(loader.dataset), score, probs_all

@torch.no_grad()
def predict_test(model, loader, device):
    model.eval()
    probs_all = []
    names_all = []

    for images, names in loader:
        images = images.to(device)
        probs_all.append(tta_probabilities(model, images).cpu().numpy())
        names_all.extend(names)

    return np.concatenate(probs_all), names_all

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working'))
    parser.add_argument('--weights', type=Path, help='Optional directory of five completed ConvNeXt checkpoints; skip retraining')
    args = parser.parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('ConvNeXt training and inference require a CUDA GPU')
    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    train_files, test_files, sample_submission = find_data(args.data)
    train_labels = np.repeat(np.arange(5), 100)
    folds = fold_ids(train_labels)
    # Keep the model summary initialization in the original training RNG sequence.
    preview = make_model()
    print(f'ConvNeXt-Tiny parameters: {sum(p.numel() for p in preview.parameters()):,}')
    del preview
    test_loader = DataLoader(
        PlantDataset(test_files, transform=valid_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(train_files), len(classes)), dtype=np.float32)
    test_probs = np.zeros((len(test_files), len(classes)), dtype=np.float32)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_files, train_labels), 1):
        print(f"\nfold {fold}/{N_FOLDS}")

        fold_train_files = [train_files[i] for i in train_idx]
        fold_valid_files = [train_files[i] for i in valid_idx]

        train_set = PlantDataset(fold_train_files, train_labels[train_idx], train_tf)
        valid_set = PlantDataset(fold_valid_files, train_labels[valid_idx], valid_tf)

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        model = make_model().to(device)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        best_score = -float("inf")
        wait = 0
        weight_path = (args.weights or output_dir) / f"convnext_fold{fold}.pth"

        for epoch in ([] if args.weights else range(1, EPOCHS + 1)):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device)
            valid_loss, valid_score, valid_probs = evaluate(model, valid_loader, loss_fn, device)
            scheduler.step()

            print(
                f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
                f"valid_loss={valid_loss:.4f}  macro_f1={valid_score:.4f}"
            )

            if valid_score > best_score:
                best_score = valid_score
                wait = 0
                torch.save(model.state_dict(), weight_path)
            else:
                wait += 1
                if wait == PATIENCE:
                    break

        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        _, best_score, valid_probs = evaluate_with_tta(model, valid_loader, loss_fn, device)
        oof_probs[valid_idx] = valid_probs
        fold_scores.append(best_score)

        fold_test_probs, test_names = predict_test(model, test_loader, device)
        test_probs += fold_test_probs / N_FOLDS

        print(f"fold {fold} best macro_f1: {best_score:.4f}")
        del model
        torch.cuda.empty_cache()

    np.savez_compressed(output_dir / 'convnext_probabilities.npz',
                        train_names=np.asarray([p.name for p in train_files]),
                        test_names=np.asarray([p.name for p in test_files]),
                        train_sha256=np.asarray([image_hash(p) for p in train_files]),
                        test_sha256=np.asarray([image_hash(p) for p in test_files]),
                        labels=train_labels, fold_id=folds,
                        convnext_oof=oof_probs, convnext_test=test_probs)
    write_submission(sample_submission, np.asarray(test_names), test_probs,
                     output_dir / 'submission_convnext.csv')
    pd.DataFrame({'fold': np.arange(1, N_FOLDS + 1),
                  'macro_f1': fold_scores}).to_csv(output_dir / 'convnext_fold_scores.csv', index=False)
    print(f'OOF accuracy: {(oof_probs.argmax(1) == train_labels).mean():.4f}')
    print(f'OOF macro-F1: {f1_score(train_labels, oof_probs.argmax(1), average="macro"):.4f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a CNN to classify characters rendered from font files.
"""

import re
import glob
import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
from PIL import Image

# ----------------------
# Dataset
# ----------------------

class CharImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, idx2char_file=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Read labels.txt: format is <file_id>\t<char>
        label_path = os.path.join(data_dir, "labels.txt")
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                file_id, char = line.strip().split('\t')
                self.samples.append((file_id, char))

        # Load char-to-index mapping from vocab file or build from samples
        if idx2char_file and os.path.exists(idx2char_file):
            self.idx2char = {}
            with open(idx2char_file, 'r', encoding='utf-8') as f:
                for line in f:
                    idx, char = line.strip().split('\t')
                    self.idx2char[int(idx)] = char
            self.char2idx = {c: i for i, c in self.idx2char.items()}
        else:
            chars = sorted(set(char for _, char in self.samples))
            self.char2idx = {c: i for i, c in enumerate(chars)}
            self.idx2char = {i: c for c, i in self.char2idx.items()}

        # Warn if unknown characters exist
        unknown_chars = {char for _, char in self.samples if char not in self.char2idx}
        if unknown_chars:
            print(f"Warning: {len(unknown_chars)} unknown characters in labels:")
            print(" ", "".join(list(unknown_chars)[:20]), "..." if len(unknown_chars) > 20 else "")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id, char = self.samples[idx]
        filename = f"{file_id}.png"
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            image = self.transform(image)
        label = self.char2idx[char]
        return image, label

# ----------------------
# Models
# ----------------------

def build_model(name, num_classes, pretrained):
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # Change input channels to 1
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # EfficientNet 输入层修改方式不同
        first_conv = model.features[0][0]
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )

        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {name}")
    return model

# ----------------------
# Train Function
# ----------------------

def train(model, dataloader, device, epochs, lr, save_path):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ===== Try to resume training =====
    last_epoch = 0
    checkpoint_files = glob.glob(os.path.join(save_path, "model_epoch*.pth"))

    if checkpoint_files:
        # 找到最大 epoch 的 checkpoint
        def extract_epoch(f):
            match = re.search(r"model_epoch(\d+)\.pth", f)
            return int(match.group(1)) if match else -1

        latest_ckpt = max(checkpoint_files, key=extract_epoch)
        last_epoch = extract_epoch(latest_ckpt)

        model.load_state_dict(torch.load(latest_ckpt, map_location=device, weights_only=True))
        print(f"Resuming from checkpoint: {latest_ckpt} (epoch {last_epoch})")

    # ===== Start training =====
    for epoch in range(last_epoch, last_epoch + epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch{epoch+1}.pth"))

# ----------------------
# Main
# ----------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 输入要求 224x224
        transforms.ToTensor(),
    ])

    dataset = CharImageDataset(
        data_dir=args.data,
        transform=transform,
        idx2char_file=args.idx2char
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --------------------------
    # Load ResNet18 and modify
    # --------------------------
    model = build_model(
        name=args.model,
        num_classes=len(dataset.char2idx),
        pretrained=args.pretrained
    )

    os.makedirs(args.save_dir, exist_ok=True)
    train(model, dataloader, device, args.epochs, args.lr, args.save_dir)

    # Save label map
    with open(os.path.join(args.save_dir, "idx2char.txt"), "w", encoding="utf-8") as f:
        for idx in sorted(dataset.idx2char):
            f.write(f"{idx}\t{dataset.idx2char[idx]}\n")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a character recognition model from font images.")
    parser.add_argument("--data", required=True, help="Path to directory with character images and labels.txt")
    parser.add_argument("--idx2char", type=str, default=None, help="Optional path to existing idx2char.txt to reuse character-to-index mapping (for consistent training/resume)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision's pretrained ResNet18 weights (on ImageNet)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0"],
        help="Model architecture to use (resnet18 or efficientnet_b0)"
    )
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save model checkpoints")

    args = parser.parse_args()
    main(args)

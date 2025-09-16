import argparse
import sys
from pathlib import Path
import logging

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import (
    densenet161,
    DenseNet161_Weights,
    densenet121,
    DenseNet121_Weights,
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    vgg19_bn,
    VGG19_BN_Weights,
)
from tqdm import tqdm


class CNNClassifier(nn.Module):
    def __init__(self, backbone_name: str = "densenet161", num_classes: int = 200):
        super().__init__()
        self.features, backbone_dim = self.get_backbone(backbone_name)
        self.classifier = nn.Linear(backbone_dim, num_classes)

    def get_backbone(self, name: str):
        assert name in ["densenet161", "densenet121", "resnet34", "resnet18", "resnet50", "vgg19"]
        if name == "densenet161":
            backbone = densenet161(weights=DenseNet161_Weights.DEFAULT)
            return nn.Sequential(
                backbone.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1)
            ), backbone.classifier.in_features
        elif name == "densenet121":
            backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
            return nn.Sequential(
                backbone.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1)
            ), backbone.classifier.in_features
        elif name == "resnet18":
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            return nn.Sequential(*list(backbone.children())[:-1]), backbone.fc.in_features
        elif name == "resnet34":
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
            return nn.Sequential(*list(backbone.children())[:-1]), backbone.fc.in_features
        elif name == "resnet50":
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            return nn.Sequential(*list(backbone.children())[:-1]), backbone.fc.in_features
        elif name == "vgg19":
            backbone = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
            return nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(start_dim=1),
                *list(backbone.classifier.children())[:3]
            ), backbone.classifier[-1].in_features

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits
    
    def predict(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        if len(self.classifier) == 2:
            concept_logits = self.classifier[0](features)
            logits = self.classifier[1](concept_logits)
        else:
            concept_logits = None
            logits = self.classifier(features)
        return logits, concept_logits


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        predicted = torch.argmax(logits, dim=-1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return train_loss / len(train_loader), correct / total


def validate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            predicted = torch.argmax(logits, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return val_loss / len(test_loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet161",
        choices=["densenet161", "densenet121", "resnet34", "resnet18", "resnet50", "vgg19"]
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    log_dir = Path(args.log_dir) / args.name
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {str(device)}")

    # Transforms
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Datasets
    train_dataset = ImageFolder(
        root=Path(args.data_dir) / "train_cropped_augmented",
        transform=transforms
    )
    test_dataset = ImageFolder(
        root=Path(args.data_dir) / "test_cropped",
        transform=transforms
    )

    num_classes = len(train_dataset.classes)
    logger.info(f"Number of classes: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Model
    model = CNNClassifier(backbone_name=args.backbone, num_classes=num_classes)
    model.to(device)

    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    best_val_acc = 0.0

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"------------------------ Epoch {epoch} ------------------------")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        logger.info(f"training loss: {train_loss:.4f}")
        logger.info(f"training acc: {train_acc:.4f}")
        logger.info(f"validation loss: {val_loss:.4f}")
        logger.info(f"validation acc: {val_acc:.4f}")

        # Checkpointing
        if val_acc > best_val_acc:
            torch.save(model.state_dict(),f"checkpoints/{args.backbone}_cub.pth")
            logger.info("Model saved as model_best.pth")
            best_val_acc = val_acc
        
        lr_scheduler.step()


if __name__ == "__main__":
    main()
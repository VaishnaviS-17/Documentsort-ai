import torch
import torch.nn as nn
from torchvision import models


def get_resnet18_model(num_classes=10, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # ResNet18 expects 3-channel (RGB) input; ours is grayscale (1 channel).
    # Modify first conv layer to accept 1 channel, initialized by averaging
    # the pretrained RGB filters.
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv1.weight = nn.Parameter(
            original_conv1.weight.mean(dim=1, keepdim=True)
        )
    model.conv1 = new_conv1

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_resnet18_model(num_classes=10, freeze_backbone=True)
    dummy_input = torch.randn(4, 1, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
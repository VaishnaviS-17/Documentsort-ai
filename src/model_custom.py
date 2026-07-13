import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A reusable conv block: Conv2d -> BatchNorm -> ReLU -> MaxPool
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DocumentCNN(nn.Module):
    """
    Custom CNN built from scratch for document type classification.
    Input: [batch, 1, 224, 224] grayscale document images
    Output: [batch, num_classes] raw logits
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.block1 = ConvBlock(1, 32)      # 224 -> 112
        self.block2 = ConvBlock(32, 64)     # 112 -> 56
        self.block3 = ConvBlock(64, 128)    # 56  -> 28
        self.block4 = ConvBlock(128, 256)   # 28  -> 14

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DocumentCNN(num_classes=10)
    dummy_input = torch.randn(4, 1, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
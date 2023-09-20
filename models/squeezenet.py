import torch
import torch.nn as nn
from torchviz import make_dot
from pytorch_model_summary import summary


class SEBlock1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
    
    def forward(self, x):
        # Global average pooling along the time dimension
        x_squeeze = self.avg_pool(x)
        # Flatten to (batch_size, in_channels)
        x_squeeze = x_squeeze.view(x_squeeze.size(0), -1)
        # Excitation operation
        x_excitation = torch.relu(self.fc1(x_squeeze))
        x_excitation = torch.sigmoid(self.fc2(x_excitation))
        # Rescale the input features
        x = x * x_excitation.unsqueeze(2)
        return x

class SEBlockResNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
        super(SEBlockResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.se_block = SEBlock1D(out_channels, reduction_ratio)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class SEResNet1D(nn.Module):
    def __init__(self, input_channels, num_classes, block, layers, reduction_ratio=16):
        super(SEResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction_ratio)
        self.layer2 = self._make_layer(block, 128, layers[1], reduction_ratio, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], reduction_ratio, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], reduction_ratio, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, reduction_ratio, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, reduction_ratio))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, reduction_ratio=reduction_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def seresnet1d(num_channels, num_classes):
    return SEResNet1D(num_channels, num_classes,
                       SEBlockResNet1D, [2, 2, 2, 2])

if __name__ == "__main__":
    # Example usage:
    num_channels = 3
    num_classes = 3
    model = SEResNet1D(num_channels, num_classes, SEBlockResNet1D, [2, 2, 2, 2])

    # model = resnet1d(3, num_classes)
    data = torch.randn(4, 3, 900)
    out = model(data)
    print(out.shape)
    print(summary(model, data, show_input=False))
    dot = make_dot(model(data), params=dict(model.named_parameters()))
    dot.format = 'png'  # You can change the format as needed
    dot.render('../model_graphs/seresnet1D_graph')
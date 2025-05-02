import torch
import torch.nn as nn
import torchvision.models as models

class Custom3DAlexNet(nn.Module):
    def __init__(self, input_size=(182, 218, 182), num_classes=1):
        super(Custom3DAlexNet, self).__init__()
        
        # AlexNet model with 3D convolutions
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=11, stride=4, padding=2),  # Conv3d instead of Conv2d
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Adjust the size of the fully connected layers after the 3D convolution
        # You need to compute the output size after the feature extraction layers
        # Using a dummy input of the same size to get the output shape after convolutions
        dummy_input = torch.zeros(1, 1, *input_size)
        feature_size = self._get_feature_size(dummy_input)

        # Replacing the fully connected layers with new sizes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(feature_size, 4096),  # Adjusted input size for FC layer
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def _get_feature_size(self, x):
        # Passing a dummy input through the feature extractor to get the output size
        x = self.features(x)
        return int(torch.prod(torch.tensor(x.size()[1:])))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
    
    class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, input_size=(182, 218, 182), num_channels=1, num_classes=1):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, stride=1, padding=1):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(growth_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.block:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

class DenseNet3D(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, growth_rate=12, block_config=(6, 12, 24, 16)):
        super(DenseNet3D, self).__init__()
        
        # Initial convolution
        self.conv0 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm3d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Dense blocks with transition layers
        num_channels = 64
        self.dense1 = DenseBlock(num_channels, growth_rate, block_config[0])
        num_channels += block_config[0] * growth_rate
        self.trans1 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense2 = DenseBlock(num_channels, growth_rate, block_config[1])
        num_channels += block_config[1] * growth_rate
        self.trans2 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense3 = DenseBlock(num_channels, growth_rate, block_config[2])
        num_channels += block_config[2] * growth_rate
        self.trans3 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense4 = DenseBlock(num_channels, growth_rate, block_config[3])
        num_channels += block_config[3] * growth_rate
        self.fc = nn.Sequential(
            nn.Linear(58350, num_channels),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, x):
        x = self.relu0(self.bn0(self.conv0(x)))
        x = self.pool0(x)

        x = self.dense1(x)
        x = self.trans1(x)

        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)

        x = self.dense4(x)
        
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

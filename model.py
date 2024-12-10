import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
#from torchvision.models import mobilenet_v2

#import torch.optim as optim
#from torchvision import models
#from torchvision.models import MobileNet_V2_Weights

# Note the model and functions here defined do not have any FL-specific components.

'''
class Net(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
'''
class Net(nn.Module):
    """A balanced CNN for CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Output: 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 32x32
        self.pool = nn.MaxPool2d(2, 2)  # Output: 16x16
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 16x16
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # Output: 16x16
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 8x8
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
'''
'''
class Net(nn.Module):
    """A deeper CNN for CIFAR-10 without Batch Normalization."""

    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        
        # Block 1: Convolutional layers with Leaky ReLU
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Output: 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 32x32
        self.pool = nn.MaxPool2d(2, 2)  # Output: 16x16
        
        # Block 2: Convolutional layers with Leaky ReLU
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 16x16
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # Output: 16x16
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 8x8
        
        # Block 3: Convolutional layers with Leaky ReLU (Optional)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output: 8x8
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: 4x4
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Output: 1x1
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = self.pool(x)
        
        # Block 2
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)
        x = self.pool2(x)
        
        # Block 3 (optional)
        x = F.leaky_relu(self.conv5(x), negative_slope=0.01)
        x = self.pool3(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)  # Flatten the output for fully connected layers
        
        # Fully connected layers with dropout
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
'''


class ResidualBlock(nn.Module):
    """
    A basic residual block without Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out += identity
        out = self.leaky_relu(out)
        return out

class Net(nn.Module):
    """
    A ResNet-inspired model for CIFAR-10 without Batch Normalization.
    """
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Initial conv
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global Average Pooling and Fully Connected Layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            )
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


'''
def train(net, trainloader, optimizer, epochs, device: str, scheduler):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    warmup_epochs = 5
    for _ in range(epochs):
        # Learning rate warm-up: Gradually increase learning rate in the first `warmup_epochs` epochs
        if epochs < warmup_epochs:
            lr = optimizer.param_groups[0]['initial_lr'] * (epochs + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    if scheduler:
        scheduler.step(epoch_loss)
'''

def train(net, trainloader, optimizer, epochs, device: str, scheduler):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    warmup_epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0.0  # Initialize epoch loss
        # Learning rate warm-up: Gradually increase learning rate in the first `warmup_epochs` epochs
        if epoch < warmup_epochs:
            lr = optimizer.param_groups[0]['initial_lr'] * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Training loop
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss for the current batch
            epoch_loss += loss.item()

        # If there is a learning rate scheduler, step it based on the epoch loss
        if scheduler:
            scheduler.step(epoch_loss)


def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def model_to_parameters(model):
    from flwr.common.parameter import ndarrays_to_parameters

    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Extracted model parameters!")
    return parameters
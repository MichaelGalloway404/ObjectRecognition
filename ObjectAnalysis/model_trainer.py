# Import necessary modules
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# Set matplotlib background color
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

# Path setup for dataset
base_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cifar10')
class_names = os.listdir(base_data_path + "/train")

# Dataset normalization stats
cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Training data transformations
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*cifar10_stats, inplace=True)
])

# Validation data transformations
valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*cifar10_stats)
])

# Load datasets
train_dataset = ImageFolder(base_data_path + '/train', train_transforms)
valid_dataset = ImageFolder(base_data_path + '/test', valid_transforms)

# DataLoader settings
batch_size = 400
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2)

# Device configuration
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def move_to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceAwareDataLoader:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def __iter__(self):
        for batch in self.data_loader:
            yield move_to_device(batch, self.device)

    def __len__(self):
        return len(self.data_loader)

# Select device and wrap DataLoaders
device = get_device()
print("Running on device:", device)
train_loader = DeviceAwareDataLoader(train_loader, device)
valid_loader = DeviceAwareDataLoader(valid_loader, device)

# Compute accuracy
def calculate_accuracy(predictions, labels):
    _, predicted_labels = torch.max(predictions, dim=1)
    return torch.tensor(torch.sum(predicted_labels == labels).item() / len(predicted_labels))

# Base class for classification models
class ClassificationModelBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        predictions = self(images)
        loss = F.cross_entropy(predictions, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        predictions = self(images)
        loss = F.cross_entropy(predictions, labels)
        accuracy = calculate_accuracy(predictions, labels)
        return {'val_loss': loss.detach(), 'val_acc': accuracy}

    def validation_epoch_end(self, results):
        losses = [x['val_loss'] for x in results]
        avg_loss = torch.stack(losses).mean()
        accuracies = [x['val_acc'] for x in results]
        avg_acc = torch.stack(accuracies).mean()
        return {'val_loss': avg_loss.item(), 'val_acc': avg_acc.item()}

    def epoch_end(self, epoch, metrics):
        print(f"Epoch [{epoch}], lr: {metrics['lrs'][-1]:.5f}, "
              f"train_loss: {metrics['train_loss']:.4f}, val_loss: {metrics['val_loss']:.4f}, "
              f"val_acc: {metrics['val_acc']:.4f}")

# Convolutional block

def conv_layer(in_channels, out_channels, apply_pooling=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if apply_pooling:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ResNet9 architecture
class ResNet9(ClassificationModelBase):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.layer1 = conv_layer(input_channels, 64)
        self.layer2 = conv_layer(64, 128, apply_pooling=True)
        self.residual_block1 = nn.Sequential(conv_layer(128, 128), conv_layer(128, 128))
        self.layer3 = conv_layer(128, 256, apply_pooling=True)
        self.layer4 = conv_layer(256, 512, apply_pooling=True)
        self.residual_block2 = nn.Sequential(conv_layer(512, 512), conv_layer(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, inputs):
        data = self.layer1(inputs)
        data = self.layer2(data)
        data = self.residual_block1(data) + data
        data = self.layer3(data)
        data = self.layer4(data)
        data = self.residual_block2(data) + data
        return self.classifier(data)

# Instantiate and move model to device
model = move_to_device(ResNet9(3, 10), device)

@torch.no_grad()
def evaluate_model(model, data_loader):
    model.eval()
    results = [model.validation_step(batch) for batch in data_loader]
    return model.validation_epoch_end(results)

def fetch_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def train_model_one_cycle(epochs, max_lr, model, train_loader, valid_loader,
                          weight_decay=0, grad_clip=None, optimizer_fn=torch.optim.SGD):
    torch.cuda.empty_cache()
    training_history = []

    optimizer = optimizer_fn(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            lrs.append(fetch_lr(optimizer))
            scheduler.step()

        metrics = evaluate_model(model, valid_loader)
        metrics['train_loss'] = torch.stack(train_losses).mean().item()
        metrics['lrs'] = lrs
        model.epoch_end(epoch, metrics)
        training_history.append(metrics)

    return training_history

# Initial evaluation
history = [evaluate_model(model, valid_loader)]

# Configuration input
use_defaults = input("Use default training parameters? (y/n): ")
if use_defaults.lower() in ('y', 'yes'):
    num_epochs = 25
    learning_rate = 0.01
else:
    num_epochs = int(input("Number of epochs [default=25]: "))
    learning_rate = float(input("Max learning rate [default=0.01]: "))

# Training parameters
gradient_clip_value = 0.1
weight_decay_value = 1e-4
optimizer_function = torch.optim.Adam

# Start training
history += train_model_one_cycle(num_epochs, learning_rate, model, train_loader, valid_loader,
                                 grad_clip=gradient_clip_value,
                                 weight_decay=weight_decay_value,
                                 optimizer_fn=optimizer_function)

def plot_validation_accuracy(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Epoch')
    plt.show()

def plot_training_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.title('Training and Validation Losses')
    plt.show()

def plot_learning_rates(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch number')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate Schedule')
    plt.show()

plot_validation_accuracy(history)
plot_training_losses(history)
plot_learning_rates(history)

def predict_class(image_tensor, model):
    image_batch = move_to_device(image_tensor.unsqueeze(0), device)
    prediction = model(image_batch)
    _, predicted_class = torch.max(prediction, dim=1)
    return train_dataset.classes[predicted_class.item()]

# Predict one image
img_tensor, true_label = valid_dataset[0]
print('Predicted label:', predict_class(img_tensor, model))

# Load a custom image for prediction
custom_img = Image.open(os.path.dirname(__file__) + '/test/test_imgs/test_img.png')
plt.imshow(np.array(custom_img))

# Save model with timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M")
model_path = os.path.join(os.path.dirname(__file__), f'model_{timestamp}.pth')
torch.save(model.state_dict(), model_path)

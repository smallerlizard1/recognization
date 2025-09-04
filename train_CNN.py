import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class RandomIMUDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=100, num_classes=6, transform=None):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.transform = transform
        self.classes = [f"class_{i}" for i in range(num_classes)]
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # 随机生成6轴IMU数据 (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
            imu_data = np.random.randn(self.seq_length, 6) * 0.1 + np.random.rand(6) * 2 - 1  # 模拟真实数据分布
            label = np.random.randint(0, self.num_classes)  # 随机标签
            data.append((imu_data, label))
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        window, label = self.data[idx]
        if self.transform:
            window = self.transform(window)
        window = torch.FloatTensor(window).permute(1, 0)  # (6, seq_length)
        return window, label


# 定义标准化函数（可选）
def normalize_data(window):
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    return (window - mean) / (std + 1e-8)

# 创建随机数据集
train_dataset = RandomIMUDataset(num_samples=1000, seq_length=100, num_classes=6, transform=normalize_data)
test_dataset = RandomIMUDataset(num_samples=200, seq_length=100, num_classes=6, transform=normalize_data)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#gpu显存有限，无法全部加载gpu，只能分批加载
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class IMU_CNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=6):
        super(IMU_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 25, 256),  # 假设seq_length=100，经过两次池化后为25
            nn.ReLU(),
            nn.Linear(256, num_classes))
        
    def forward(self, x):
        x = self.conv1(x)  # (B, 64, 50)
        x = self.conv2(x)  # (B, 128, 25)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IMU_CNN(input_channels=6, num_classes=len(train_dataset.classes)).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, epochs=10):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试集评估
        test_loss, test_acc = evaluate_model(model, test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    return train_losses, test_losses, train_accs, test_accs

def evaluate_model(model, dataloader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100 * correct / total

# 训练模型
train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, epochs=20)



plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.legend()
plt.title('Accuracy Curve')
plt.show()



def print_classification_report(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=train_dataset.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

print_classification_report(model, test_loader)
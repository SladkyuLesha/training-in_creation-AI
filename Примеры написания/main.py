import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Загрузка данных
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 2. Модель
class SimpleNet(nn.Module):
    def __init__(self, activation='relu'):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = activation

    def forward(self, x):
        x = x.view(-1, 28*28)
        if self.activation == 'relu':
            x = F.relu(self.fc1(x))
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(self.fc1(x))
        elif self.activation == 'tanh':
            x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Обучение
def train(model, loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 4. Запуск
model = SimpleNet(activation='tanh')  # попробуй 'sigmoid' и 'tanh'
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print('tanh')
train(model, train_loader, optimizer, criterion, epochs=3)
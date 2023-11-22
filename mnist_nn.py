import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Définition du modèle NN
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Second test avec un réseau plus complexe -> 1er exec : 97.42% d'accuracy | 2eme exec : 96.31% d'accuracy
        """
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        """

        #Premier test avec un réseau simple -> 1er exec 97.42% d'accuracy | 2eme exec : 97.32% d'accuracy
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Chargement du dataset MNIST
print("Chargement du dataset MNIST...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instanciation du modèle, définition de la fonction de perte et de l'optimiseur
print("Initialisation du modèle et de l'optimiseur...")
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
print("Début de l'entraînement...")
for epoch in range(10):  # nombre d'époques
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Époque {epoch+1}/{10}, Batch {batch_idx+1}/{len(train_loader)}, Perte: {loss.item():.6f}")

    print(f"Époque {epoch+1}, Perte moyenne: {total_loss / len(train_loader):.6f}")

# Évaluation du modèle
print("Début de l'évaluation...")
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Précision du réseau sur 10 000 tests : {100 * correct / total}%')

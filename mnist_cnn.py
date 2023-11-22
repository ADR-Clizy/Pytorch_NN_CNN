import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Définition du modèle CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Test du réseau CNN -> 1er exec 99.0% d'accuracy | 2eme exec : 99.2% d'accuracy
        # Première couche de convolution : 1 entrée (28x28), 32 sorties, kernel de 3x3, déplacement de 1, padding de 1 pour les bords
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Deuxième couche de convolution : 32 entrées (14x14), 64 sorties, kernel de 3x3, déplacement de 1, padding de 1 pour les bords
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Couche linéaire : 7x7x64 entrées, 128 sorties
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        #Extraction des meilleures caractéristiques de l'image et réduction de la taille de l'image par 2
        #Sortie -> 14x14x32
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        #Sortie -> 7x7x64
        x = torch.max_pool2d(x, 2)
        #Aplatir les données pour les passer dans le réseau de neurones en sortie des couches de convolution
        x = x.view(-1, 7*7*64)
        #Appliquer le relu sur la couche linéaire
        x = torch.relu(self.fc1(x))
        #Affecter à x la sortie de la dernière couche linéaire
        x = self.fc2(x)
        return x

# Chargement du dataset MNIST
print("Chargement du dataset MNIST...")
# Normalisation des données pour avoir des valeurs entre -1 et 1 pour chaque pixel
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Initialisation du modèle et de l'optimiseur...")
# Instanciation du modèle, définition de la fonction de perte et de l'optimiseur
model = SimpleCNN()
# Definition de la fonction de perte
criterion = nn.CrossEntropyLoss()
# Definition de l'optimiseur (Pour la reduction de la perte)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Début de l'entraînement...")
# Entraînement du modèle
for epoch in range(10):  # nombre d'époques
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Remettre à zéro les gradients car sinon ils s'accumulent à chaque itération
        # On veut ici calculer les gradients pour chaque batch
        optimizer.zero_grad()
        # Calculer la sortie du modèle
        output = model(data)
        # Calculer la perte
        loss = criterion(output, target)
        # Calculer les gradients
        loss.backward()
        # Mettre à jour les poids
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
        # Calculer la sortie du modèle
        output = model(data)
        # Récupérer la valeur maximale de la sortie du modèle
        _, predicted = torch.max(output.data, 1)
        # Calculer le nombre total d'images
        total += target.size(0)
        # Calculer le nombre d'images correctement prédites
        correct += (predicted == target).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

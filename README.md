# Projet MNIST avec PyTorch

Ce projet implémente deux modèles différents pour classer les chiffres écrits à la main du dataset MNIST en utilisant PyTorch : un modèle de réseau de neurones simple (NN) et un modèle de réseau de neurones convolutif (CNN).

## Prérequis

- Python 3.6 ou supérieur
- PyTorch
- Torchvision

## Installation

1. **Clonez le dépôt** : 
   ```bash
   git clone [URL_DU_DÉPÔT]
   cd [NOM_DU_DÉPÔT]
   ```

2. **Créez et activez un environnement virtuel (optionnel)** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. **Installez les dépendances** :
   ```bash
   pip install torch torchvision
   ```

## Utilisation

Pour exécuter les modèles, lancez les scripts Python correspondants.

1. **Pour le modèle de réseau de neurones simple (NN)** : 
   ```bash
   python mnist_nn.py
   ```
2. **Pour le modèle de réseau de neurones convolutif (CNN)** : 
   ```bash
   python mnist_cnn.py
   ```

## Structure du Projet

- `mnist_nn.py` : Script pour le modèle de réseau de neurones simple (NN).
- `mnist_cnn.py` : Script pour le modèle de réseau de neurones convolutif (CNN).

## Résultats Attendus

Après l'exécution des scripts, vous verrez la progression de l'entraînement et la précision du modèle sur l'ensemble de test. Les résultats typiques sont les suivants :
- Précision pour le modèle NN : Environ 97%
- Précision pour le modèle CNN : Environ 99%

## Auteurs

- LIZY Corentin
- VIALE Alexandre



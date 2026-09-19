import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Pipeline de traitement de données (Biais Inductif Standard)
# On convertit les images en Tenseurs PyTorch et on les normalise entre -1 et 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Téléchargement automatique du dataset MNIST (60 000 images d'entraînement de taille 28x28)
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 2. Définition de l'Architecture de notre premier CNN
class PremierCNN(nn.Module):
    def __init__(self):
        super(PremierCNN, self).__init__()
        # Couche Convolutive 1 : Prend 1 canal (Niveaux de gris), sort 16 cartes, filtre 3x3, padding 1
        # Entrée : (64, 1, 28, 28) -> Sortie : (64, 16, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        # Max Pooling : Divise la hauteur et la largeur par 2 en ne gardant que le pixel max d'une zone 2x2
        # Entrée : (64, 16, 28, 28) -> Sortie : (64, 16, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Couche Fully Connected (MLP de sortie) : On aplatit les 16 cartes de taille 14x14
        self.fc = nn.Linear(16 * 14 * 14, 10) # 10 classes de sortie (chiffres de 0 à 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Étape d'aplatissement (Flattening) pour passer du monde spatial au monde linéaire
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

# 3. Initialisation du Matériel et des Hyperparamètres
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- INITIALISATION MATÉRIELLE : {str(device).upper()} ---")

modele = PremierCNN().to(device)
critere = nn.CrossEntropyLoss() # Inclut nativement la Softmax et la Log Loss multi-classe
optimiseur = optim.Adam(modele.parameters(), lr=0.001) # L'optimiseur adaptatif que nous avons étudié !

# 4. Boucle d'entraînement ultra-rapide (1 seule époque pour valider l'infrastructure)
print("\nDébut de l'entraînement sur GPU...")
modele.train()
for epoch in range(1):
    running_loss = 0.0
    for idx, (images, labels) in enumerate(trainloader):
        # Envoi direct des matrices de données sur la VRAM du GPU
        images, labels = images.to(device), labels.to(device)
        
        # Remise à zéro des gradients accumulés à l'étape précédente
        optimiseur.zero_grad()
        
        # Forward pass
        sorties = modele(images)
        loss = critere(sorties, labels)
        
        # Backward pass automatique (Autograd de PyTorch)
        loss.backward()
        
        # Mise à jour des poids via les équations d'Adam
        optimiseur.step()
        
        running_loss += loss.item()
        if idx % 200 == 199:
            print(f"Batch {idx+1}/{len(trainloader)} -> Erreur (Loss): {running_loss / 200:.4f}")
            running_loss = 0.0

print("\n--- Entraînement terminé ! ---")

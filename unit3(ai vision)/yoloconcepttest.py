import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. GÉNÉRATION DES DONNÉES SIMULÉES
layer_20_output = torch.zeros((1, 2, 4, 4))
layer_20_output[0, 0, 3, 1] = 1.0  # Roue avant
layer_20_output[0, 0, 3, 3] = 1.0  # Roue arrière
layer_20_output[0, 1, 2, 1:4] = 1.0 # Carrosserie juste au-dessus

# 2. DESIGN DE LA COUCHE DE CONCEPT DE HAUT NIVEAU
class DeepConceptLayer(nn.Module):
    def __init__(self):
        super(DeepConceptLayer, self).__init__()
        self.conv_concept = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(2, 3), bias=False)
        
        # Poids géométriques appris pour assembler la voiture
        concept_weights = torch.zeros((1, 2, 2, 3))
        concept_weights[0, 0, 1, 0] = 2.0  # Cible la roue gauche
        concept_weights[0, 0, 1, 2] = 2.0  # Cible la roue droite
        concept_weights[0, 1, 0, :] = 1.0  # Cible la carrosserie au-dessus
        
        self.conv_concept.weight = nn.Parameter(concept_weights, requires_grad=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv_concept(x))

# 3. EXÉCUTION DU MODÈLE
model = DeepConceptLayer()
with torch.no_grad():
    concept_activation = model(layer_20_output)

# --- 4. CONVERSION EN GRAPHIQUE VISUEL ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Affichage du Canal 0 (Roues)
im0 = axes[0].imshow(layer_20_output[0, 0].numpy(), cmap='coolwarm', vmin=0, vmax=1)
axes[0].set_title("Canal 0 : Détection des Roues (Bas niveau)")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Affichage du Canal 1 (Carrosserie)
im1 = axes[1].imshow(layer_20_output[0, 1].numpy(), cmap='coolwarm', vmin=0, vmax=1)
axes[1].set_title("Canal 1 : Détection Carrosserie (Bas niveau)")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Affichage de la Sortie (Concept Voiture)
# La taille est réduite à 3x2 à cause de la taille du filtre sans padding
im2 = axes[2].imshow(concept_activation[0, 0].numpy(), cmap='hot', vmin=0, vmax=7)
axes[2].set_title("Couche 30 : Activation du Concept 'Voiture'")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

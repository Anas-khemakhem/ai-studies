import numpy as np

# 1. PARAMÈTRES INITIALISATION
# On imagine qu'on optimise une seule variable w (ex: un poids ou une coordonnée)
w = 10.0          # Position initiale
v = 0.0           # Vitesse initiale (toujours à 0 au départ)

# Hyperparamètres
beta = 0.9        # Coefficient de friction (conserve 90% de la vitesse précédente)
lr = 0.1          # Learning rate (η)

print(f"Départ -> Position w: {w:.4f} | Vitesse v: {v:.4f}\n")

# 2. SIMULATION DE LA BOUCLE D'OPTIMISATION
# On simule 5 étapes où la pente (le gradient dw) vaut toujours 2.0
dw = 2.0 

for t in range(1, 6):
    # Équation 1 : Mise à jour de la vitesse (Inertie + Nouvelle pente)
    v = beta * v + (1.0 - beta) * dw
    
    # Équation 2 : Mise à jour de la position (Déplacement)
    w -= lr * v
    
    print(f"Étape {t} -> Vitesse v: {v:.4f} | Position w: {w:.4f}")

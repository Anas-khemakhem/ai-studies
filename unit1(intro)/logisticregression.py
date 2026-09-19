import numpy as np

# 1. Données réelles simulées : X = [Classe, Prix du billet]
X = np.array([
    [3, 7.25],   # Passager 1 : 3ème classe, tarif bas
    [1, 71.28],  # Passager 2 : 1ère classe, tarif élevé
    [3, 7.92],   # Passager 3 : 3ème classe, tarif bas
    [1, 53.10],  # Passager 4 : 1ère classe, tarif élevé
    [2, 13.00]   # Passager 5 : 2ème classe, tarif moyen
])

# Y = [0, 1, 1, 1, 0] -> Statut de survie (0 = Mort, 1 = Vivant)
# On le transforme en matrice colonne pour aligner les dimensions avec nos poids
Y = np.array([[0], [1], [1], [1], [0]])

# 2. Initialisation des poids (2 caractéristiques = 2 poids dans notre vecteur W)
W = np.array([[0.0], [0.0]]) 
b = 0.0

# 3. Paramètres
learning_rate = 0.01
epochs = 5000
n = len(X)

# Définition de notre fonction d'activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. Boucle d'apprentissage
for i in range(epochs):
    
    # --- FORWARD PASS ---
    # Produit matriciel entre X (5, 2) et W (2, 1) -> donne une matrice (5, 1)
    z = np.dot(X, W) + b
    Y_pred = sigmoid(z)
    
    # Calcul de l'erreur (Log Loss / Binary Cross Entropy)
    # np.clip évite les erreurs mathématiques de log(0)
    Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))
    
    # --- BACKWARD PASS (Gradients) ---
    # X.T est la transposée de X, indispensable pour la multiplication matricielle des gradients
    dW = (1/n) * np.dot(X.T, (Y_pred - Y))
    db = (1/n) * np.sum(Y_pred - Y)
    
    # --- UPDATE ---
    W = W - (learning_rate * dW)
    b = b - (learning_rate * db)
    
    if i % 1000 == 0:
        print(f"Étape {i} -> Erreur: {loss:.4f}")

print("\n--- Entraînement terminé ! ---")

# 5. TEST DE PREDICTION
# Un nouveau passager arrive : 1ère classe, billet à 80.00
nouveau_passager = np.array([[1, 80.00]])
score_brut = np.dot(nouveau_passager, W) + b
probabilite = sigmoid(score_brut)

print(f"Probabilité de survie calculée par ton IA : {probabilite[0][0] * 100:.2f}%")

import numpy as np

# 1. Nos données "expérimentales" (Vitesse vs Distance de freinage)
X = np.array([10, 20, 30, 40, 50])  # Vitesse en km/h
Y = np.array([5, 12, 22, 35, 50])   # Distance en mètres

# 2. Initialisation des paramètres de l'IA au hasard
W = 0.0  # Un poids pour chaque colonne 
b = 0.0

# 3. Hyperparamètres (réglages de l'entraînement)
learning_rate = 0.0001
epochs = 1000  # Nombre de fois où l'IA va revoir les données
n = len(X)     # Nombre d'échantillons (ici 5)

# 4. Boucle d'apprentissage (Gradient Descent)
for i in range(epochs):
    
    # --- ÉTAPE A : FORWARD PASS ---
    # Calcule la prédiction brute (Y_pred = X * W + b)
    Y_pred = X*W+b
    
    # Calcul de l'erreur (Mean Squared Error)
    loss = np.mean((Y_pred - Y) ** 2)
    
    # --- ÉTAPE B : CALCUL DES GRADIENTS (DÉRIVÉES) ---
    # Traduis les formules mathématiques des dérivées en code Python
    dW = (2/n) * np.sum(X * (Y_pred - Y))
    db = (2/n) * np.sum(Y_pred - Y)
    
    # --- ÉTAPE C : UPDATE DES POIDS ---
    # Soustrais le gradient (multiplié par le learning_rate) à W et b
    W = W - (learning_rate * dW)
    b = b - (learning_rate * db)
    
    # On affiche l'évolution de l'erreur toutes les 100 étapes
    if i % 100 == 0:
        print(f"Étape {i} -> Erreur (Loss): {loss:.4f} | W: {W:.4f} | b: {b:.4f}")

# 5. TEST DE TON IA
print("\n--- Entraînement terminé ! ---")
vitesse_test = 60
distance_predite = vitesse_test * W + b
print(f"Pour une vitesse de {vitesse_test} km/h, l'IA prédit une distance de : {distance_predite:.2f} mètres")

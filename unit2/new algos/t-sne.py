import numpy as np

# --- 1. ESPACE DE HAUTE DIMENSION (Ton code d'origine) ---
def compute_high_dim_probabilities(X):
    N, D = X.shape
    sum_X = np.sum(X**2, axis=1, keepdims=True)
    distances_squared = sum_X + sum_X.T - 2 * np.dot(X, X.T)
    
    exponent_matrix = np.exp(-distances_squared / 2.0)
    np.fill_diagonal(exponent_matrix, 0)
    
    row_sums = np.sum(exponent_matrix, axis=1, keepdims=True)
    p_conditional = exponent_matrix / (row_sums + 1e-15)
    
    P = (p_conditional + p_conditional.T) / (2 * N)
    return np.maximum(P, 1e-12)

# --- 2. ESPACE DE FAIBLE DIMENSION (La carte en 2D avec Student-t) ---
def compute_low_dim_probabilities(Y):
    N, _ = Y.shape
    sum_Y = np.sum(Y**2, axis=1, keepdims=True)
    distances_squared_2d = sum_Y + sum_Y.T - 2 * np.dot(Y, Y.T)
    
    # Distribution de Student (t-distribution avec v=1)
    inv_distances = 1.0 / (1.0 + distances_squared_2d)
    np.fill_diagonal(inv_distances, 0)
    
    Q = inv_distances / np.sum(inv_distances)
    return np.maximum(Q, 1e-12), inv_distances

# --- 3. L'ARBITRE : COÛT ET GRADIENT (Kullback-Leibler) ---
def compute_kl_loss_and_gradient(P, Q, Y, inv_distances):
    N, no_dims = Y.shape
    
    # Calcul de la perte KL
    loss = np.sum(P * np.log(P / Q))
    
    # Calcul du gradient (indique dans quelle direction bouger les points 2D)
    dY = np.zeros((N, no_dims))
    PQ_diff = P - Q
    for i in range(N):
        # Version mathématique simplifiée du gradient du t-SNE
        direction = (PQ_diff[i, :].reshape(-1, 1) * inv_distances[i, :].reshape(-1, 1) * (Y[i, :] - Y))
        dY[i, :] = 4.0 * np.sum(direction, axis=0)
        
    return loss, dY

# --- 4. PIPELINE D'ENTRAÎNEMENT ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # Vos données d'origine (Point 0 et 1 proches, Point 2 très loin)
    X_test = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0]
    ])
    
    # Étape A : On calcule la matrice des cibles (P)
    P = compute_high_dim_probabilities(X_test)
    
    # Étape B : On place les points au hasard sur notre feuille 2D pour commencer
    Y_2d = np.random.randn(3, 2) * 1e-4
    
    # Étape C : Paramètres de la descente de gradient
    learning_rate = 10.0
    epochs = 500
    
    print("--- DÉBUT DE L'OPTIMISATION DU t-SNE ---\n")
    
    for epoch in range(epochs):
        # 1. Calculer la carte actuelle en 2D (Q)
        Q, inv_distances = compute_low_dim_probabilities(Y_2d)
        
        # 2. Calculer l'erreur (KL) et la direction des mouvements (Gradient)
        loss, dY = compute_kl_loss_and_gradient(P, Q, Y_2d, inv_distances)
        
        # 3. Mettre à jour la position des points en 2D (Descente de gradient)
        Y_2d -= learning_rate * dY
        
        # Affichage du suivi toutes les 100 étapes
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Étape {epoch+1:03d} | Erreur KL (Loss): {loss:.6f}")
            
    print("\n--- POSITIONS FINALES EN 2D ---")
    for i in range(len(Y_2d)):
        print(f"Point {i} : [{Y_2d[i, 0]:.4f}, {Y_2d[i, 1]:.4f}]")

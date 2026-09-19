import numpy as np

# --- 1. ESPACE DE HAUTE DIMENSION (Graphe des plus proches voisins) ---
def compute_umap_high_dim(X, n_neighbors=2):
    N, D = X.shape
    sum_X = np.sum(X**2, axis=1, keepdims=True)
    distances_squared = np.maximum(sum_X + sum_X.T - 2 * np.dot(X, X.T), 0)
    distances = np.sqrt(distances_squared)
    
    # Pour chaque point, on cherche la distance au plus proche voisin (rhos)
    # On trie les distances par ligne
    sorted_distances = np.sort(distances, axis=1)
    # rho_i est la distance au tout premier voisin (excluant le point lui-même qui est à 0)
    rhos = sorted_distances[:, 1].reshape(-1, 1)
    
    # Dans un vrai UMAP, sigma est calculé pour que la somme des poids locaux soit log2(n_neighbors)
    # Pour notre exemple simplifié, on fixe sigma de manière stable
    sigmas = np.ones((N, 1))
    
    # Calcul des forces d'amitié asymétriques
    # On soustrait rho pour s'assurer que le voisin le plus proche a toujours un poids de 1.0 (local connectivity)
    P_directional = np.exp(-np.maximum(distances - rhos, 0) / sigmas)
    np.fill_diagonal(P_directional, 0)
    
    # UMAP combine les probabilités avec une intersection floue (Fuzzy Union)
    # Formule : P = P + P.T - P * P.T
    P = P_directional + P_directional.T - (P_directional * P_directional.T)
    return P

# --- 2. ESPACE DE FAIBLE DIMENSION (Courbe UMAP spécifique) ---
def compute_umap_low_dim(Y, a=1.577, b=0.895):
    # Les constantes 'a' et 'b' sont les valeurs par défaut d'UMAP pour reproduire une loi de Student modifiée
    N, _ = Y.shape
    sum_Y = np.sum(Y**2, axis=1, keepdims=True)
    distances_squared_2d = np.maximum(sum_Y + sum_Y.T - 2 * np.dot(Y, Y.T), 0)
    
    # Formule UMAP basse dimension : 1 / (1 + a * d^(2b))
    Q = 1.0 / (1.0 + a * (distances_squared_2d ** b))
    np.fill_diagonal(Q, 0)
    return Q

# --- 3. L'ARBITRE UMAP : ENTROPIE CROISÉE FLOUE ---
def compute_umap_loss_and_gradient(P, Q, Y, a=1.577, b=0.895):
    N, no_dims = Y.shape
    eps = 1e-12
    
    # 1. Calcul du Coût (Fuzzy Cross-Entropy)
    # UMAP ne regarde pas seulement ce qui doit être proche (comme le t-SNE), 
    # il pousse aussi activement ce qui doit être éloigné (le terme avec 1-P)
    loss = -np.sum(P * np.log(Q + eps) + (1.0 - P) * np.log(1.0 - Q + eps))
    
    # 2. Calcul du Gradient vectorisé (forces d'attraction et de répulsion)
    dY = np.zeros((N, no_dims))
    sum_Y = np.sum(Y**2, axis=1, keepdims=True)
    dist_sq = np.maximum(sum_Y + sum_Y.T - 2 * np.dot(Y, Y.T), 0)
    dist = np.sqrt(dist_sq) + eps
    
    for i in range(N):
        # Force attractive (quand P > 0)
        attr_force = P[i, :] * (b * a * (dist_sq[i, :] ** (b - 1))) / (1.0 + a * (dist_sq[i, :] ** b) + eps)
        # Force répulsive (quand P est proche de 0, pousse les points éloignés)
        rep_force = ((1.0 - P[i, :]) * b) / (dist_sq[i, :] * (1.0 + a * (dist_sq[i, :] ** b)) + eps)
        
        # Le sens du déplacement dépend de la combinaison des deux forces
        total_force = attr_force - rep_force
        direction = (total_force.reshape(-1, 1) * (Y[i, :] - Y))
        dY[i, :] = np.sum(direction, axis=0)
        
    return loss, dY

# --- 4. PIPELINE D'ENTRAÎNEMENT ---
if __name__ == "__main__":
    np.random.seed(42)
    
    # Même jeu de données (Point 0 et 1 proches, Point 2 très loin)
    X_test = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0]
    ])
    
    # Étape A : Matrice haute dimension UMAP
    P = compute_umap_high_dim(X_test, n_neighbors=2)
    
    # Étape B : Initialisation des points en 2D (UMAP utilise souvent une ACP/PCA pour démarrer, ici random pour l'exemple)
    Y_2d = np.random.randn(3, 2) * 0.1
    
    # Étape C : Boucle d'optimisation
    learning_rate = 1.0
    epochs = 300
    
    print("--- DÉBUT DE L'OPTIMISATION UMAP ---\n")
    
    for epoch in range(epochs):
        Q = compute_umap_low_dim(Y_2d)
        loss, dY = compute_umap_loss_and_gradient(P, Q, Y_2d)
        
        # Descente de gradient
        Y_2d -= learning_rate * dY
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Étape {epoch+1:03d} | Fuzzy Cross-Entropy Loss : {loss:.6f}")
            
    print("\n--- POSITIONS FINALES UMAP EN 2D ---")
    for i in range(len(Y_2d)):
        print(f"Point {i} : [{Y_2d[i, 0]:.4f}, {Y_2d[i, 1]:.4f}]")

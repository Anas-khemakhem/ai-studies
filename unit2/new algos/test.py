import numpy as np

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # N = nombre d'échantillons, D = nombre de caractéristiques
        N, D = X.shape
        
        # CHALLENGE 1 : Centre les données (Soustrais la moyenne de chaque colonne)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calcul SVD efficace : On utilise 'full_matrices=False' pour économiser la RAM
        # U: (N, K), S: (K,), Vt: (K, D) où K = min(N, D)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Les composantes principales sont les lignes de Vt (ou colonnes de V)
        # On ne garde que les 'n_components' premières lignes
        self.components = Vt[:self.n_components] # Taille (n_components, D)
        
        # CHALLENGE 2 : Calcule le ratio de variance expliquée
        # En SVD, les valeurs propres de la matrice de covariance sont égales à : (S^2) / (N - 1)
        eigenvalues = (S ** 2) / (N - 1)
        
        # Ratio : valeurs propres sélectionnées divisées par la somme de TOUTES les valeurs propres
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

    def transform(self, X):
        # CHALLENGE 3 : Centre les nouvelles données de test X avec la moyenne apprise au 'fit'
        X_centered = X - self.mean
        
        # CHALLENGE 4 : Projette les données centrées sur les composantes principales
        # (N, D) @ (D, n_components) -> (N, n_components)
        X_projected = X_centered @ self.components.T
        
        return X_projected

# --- SCRIPT DE VÉRIFICATION AUTOMATIQUE ---
if __name__ == "__main__":
    np.random.seed(42)
    # 100 points dans un espace à 5 dimensions
    Faux_Data = np.random.randn(100, 5)
    
    pca = PCAFromScratch(n_components=2)
    pca.fit(Faux_Data)
    Data_Reduit = pca.transform(Faux_Data)
    
    print("--- RAPPORT DE COMPILATION ---")
    print(f"Forme finale après réduction (doit être (100, 2)) : {Data_Reduit.shape}")
    print(f"Variance totale conservée sur les 2 axes : {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

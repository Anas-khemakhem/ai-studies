import numpy as np

class KNNFromScratch:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y):
        # Le KNN n'apprend rien, il stocke les données en mémoire
        self.X_train = X
        self.y_train = y
        
    def predict(self, X_test):
        # X_test est une matrice de taille (M, D)
        # self.X_train est une matrice de taille (N, D)
        
        if self.metric == 'euclidean':
            # Astuce IOI : Calcul matriciel des distances sans aucune boucle
            # (A - B)^2 = A^2 + B^2 - 2AB
            X_test_square = np.sum(X_test**2, axis=1, keepdims=True) # (M, 1)
            X_train_square = np.sum(self.X_train**2, axis=1) # (N,)
            dot_product = np.dot(X_test, self.X_train.T) # (M, N)
            
            # Distances de taille (M, N)
            distances = np.sqrt(X_test_square + X_train_square - 2 * dot_product)
            
        elif self.metric == 'cosine':
            # Normalisation L2 des vecteurs pour le calcul du cosinus
            norm_test = np.linalg.norm(X_test, axis=1, keepdims=True)
            norm_train = np.linalg.norm(self.X_train, axis=1, keepdims=True)
            
            dot_product = np.dot(X_test, self.X_train.T)
            distances = 1 - (dot_product / (np.dot(norm_test, norm_train.T) + 1e-15))
            
        # Tri des indices des K plus petites distances pour chaque point de test
        # np.argsort sur l'axis 1 trie chaque ligne indépendamment : Complexité O(M * N log N)
        k_indices = np.argsort(distances, axis=1)[:, :self.k] # (M, K)
        
        # Récupération des étiquettes des voisins
        k_labels = self.y_train[k_indices] # (M, K)
        
        # Vote majoritaire pour chaque ligne
        predictions = []
        for labels in k_labels:
            counts = np.bincount(labels)
            predictions.append(np.argmax(counts))
            
        return np.array(predictions)

# --- ZONE DE TEST ACCÉLÉRÉE ---
if __name__ == "__main__":
    X_tr = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 7.0]])
    y_tr = np.array([0, 0, 1, 1]) # Deux clusters distincts (0 et 1)
    
    X_te = np.array([[1.2, 1.9], [5.5, 7.5]])
    
    knn = KNNFromScratch(k=3, metric='euclidean')
    knn.fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    print("Prédictions NumPy de zéro (doit être) :", preds)

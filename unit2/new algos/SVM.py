import numpy as np

class SVMFromScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param # Paramètre de régularisation (1/C)
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # S'assurer que les étiquettes y sont strictly -1 ou 1
        y_transformed = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        # Initialisation des poids
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condition de la Hinge Loss : condition KKT
                condition = y_transformed[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Le point est bien classé au-delà de la marge.
                    # Seule la régularisation L2 applique un gradient sur w.
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Le point viole la marge ou est mal classé. On applique le gradient d'erreur.
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_transformed[idx]))
                    self.b -= self.lr * (-y_transformed[idx])

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx) # Renvoie -1 ou 1

# --- TEST ---
if __name__ == "__main__":
    X = np.array([[2, 3], [1, 1], [2, 1], [4, 5], [5, 6], [4, 4]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    
    svm = SVMFromScratch(n_iters=500)
    svm.fit(X, y)
    print("Poids finaux w du SVM :", svm.w)
    print("Biais final b du SVM :", svm.b)

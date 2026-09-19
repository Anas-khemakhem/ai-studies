import numpy as np

class MLP:
    def __init__(self, d_in, h, d_out):
        # Initialisation He (He Initialization), standard pour ReLU
        self.W1 = np.random.randn(d_in, h) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros((1, h))
        self.W2 = np.random.randn(h, d_out) * np.sqrt(2.0 / h)
        self.b2 = np.zeros((1, d_out))
        
    def relu(self, z):
        return np.maximum(0, z)
        
    def relu_derivee(self, z):
        return (z > 0).astype(float)
        
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
    def train(self, X, Y, epochs=10000, lr=0.1):
        N = X.shape[0]
        
        for i in range(epochs):
            # 1. FORWARD PASS
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.relu(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            Y_pred = self.softmax(Z2)
            
            # Calcul de la Loss (Cross-Entropy)
            loss = -np.sum(Y * np.log(Y_pred + 1e-15)) / N
            
            # 2. BACKWARD PASS (Rétropropagation)
            # Étape A : Erreur de sortie d2
            d2 = Y_pred - Y
            
            # Étape B : Gradients de la couche 2
            dW2 = (1 / N) * np.dot(A1.T, d2)
            db2 = (1 / N) * np.sum(d2, axis=0, keepdims=True)
            
            # Étape C : Erreur cachée d1
            # Aide : utilise self.relu_derivee(Z1) et l'opérateur * pour le produit de Hadamard
            d1 = np.dot(d2, self.W2.T) * self.relu_derivee(Z1) 
            
            # Étape D : Gradients de la couche 1
            dW1 = (1 / N) * np.dot(X.T, d1)
            db1 = (1 / N) * np.sum(d1, axis=0, keepdims=True)
            
            # 3. MISE À JOUR DES PARAMÈTRES (Gradient Descent)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            
            if i % 1000 == 0:
                print(f"Étape {i} -> Loss: {loss:.4f}")

# --- TEST SUR LE PROBLÈME DU XOR ---
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
# Y en One-Hot Encoding : [Classe 0, Classe 1]
Y_xor = np.array([[1,0], [0,1], [0,1], [1,0]])

# Initialisation d'un MLP : 2 entrées, 4 neurones cachés, 2 sorties
mon_mlp = MLP(d_in=2, h=4, d_out=2)
print("Entraînement du MLP sur le XOR...")
mon_mlp.train(X_xor, Y_xor, epochs=11000, lr=0.1)

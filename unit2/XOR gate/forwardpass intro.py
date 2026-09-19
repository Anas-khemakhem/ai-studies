import numpy as np

# Fixons les graines aléatoires pour avoir les mêmes résultats
np.random.seed(42)

# 1. Génération de fausses données (Simulons 5 passagers avec 4 caractéristiques)
N = 5
D_in = 4
H = 3        # On choisit 3 neurones cachés pour l'exemple
D_out = 2    # 2 classes de sortie (0 ou 1)

X = np.random.randn(N, D_in) # Matrice (5, 4)

# 2. Initialisation des poids (On utilise une distribution normale)
W1 = np.random.randn(D_in, H)   # Taille (4, 3)
b1 = np.zeros((1, H))           # Taille (1, 3)

W2 = np.random.randn(H, D_out)  # Taille (3, 2)
b2 = np.zeros((1, D_out))       # Taille (1, 2)

# --- FONCTIONS D'ACTIVATION À CODER ---
def relu(z):
    # Aide : np.maximum(0, z) fait un max élément par élément
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- TON TRAVAIL : LE FORWARD PASS ---
# Étape A : Calcule Z1 et A1
Z1 = np.dot(X, W1) + b1
A1 = relu(Z1)

# Étape B : Calcule Z2 et Y_pred
Z2 =A1@W2 + b2 # À TOI DE COMPLÉTER ICI (Produit matriciel entre A1 et W2 + b2)
Y_pred =softmax(Z2) # À TOI DE COMPLÉTER ICI (Passe Z2 dans la fonction softmax)

print("Forme de Y_pred (doit être (5, 2)) :", Y_pred.shape)
print("\nMatrice de prédictions finales (Probabilités pour chaque classe) :\n", Y_pred)

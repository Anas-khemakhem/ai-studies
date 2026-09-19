import numpy as np

def softmax(Z):
    # Z est une matrice de scores bruts (par exemple de taille 5 lignes pour 5 passagers, 3 colonnes pour 3 classes)
    # Étape 1 : Trouve le maximum de chaque ligne pour éviter l'overflow
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    
    # Étape 2 : Divise chaque élément par la somme de sa ligne
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

#remarque
#Avec axis=1 seul Résultat : [2.0, 3.0] 
#/Avec axis=1, keepdims=True Résultat : [[2.0], [3.0]] 

# --- TEST DE TA FONCTION ---
# Imaginons les scores bruts de 2 passagers pour 3 classes de cabines possibles (A, B ou C)
scores_bruts = np.array([
    [2.0, 1.0, 0.1],  # Passager 1 (il semble préférer la classe A)
    [1.0, 3.0, 0.2]   # Passager 2 (il semble préférer la classe B)
])

probabilites = softmax(scores_bruts)
print("Probabilités par classe pour chaque passager :\n", probabilites)
print("\nVérification (la somme de chaque ligne doit valoir 1) :", np.sum(probabilites, axis=1))

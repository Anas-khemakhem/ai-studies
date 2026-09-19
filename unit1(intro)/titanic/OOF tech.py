import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# --- PRÉREQUIS : OBTENIR LES PROBABILITÉS SUR LE TRAIN ---
# Imaginons que tu as entraîné ta Régression Logistique (NumPy) et ta Random Forest (Sklearn)

# 1. Génération des probabilités sur le jeu d'entraînement (Train)
# (Pour ton code NumPy, c'est ton vecteur Y_pred obtenu à la dernière itération)
proba_train_log = Y_pred  # Forme (891, 1)

# (Pour Sklearn, on utilise .predict_proba() qui renvoie [proba_mort, proba_survie])
# On ne garde que la colonne de survie index 1
# Supposons que 'rf_model' est ton modèle de forêt déjà entraîné
proba_train_rf = rf_model.predict_proba(X_train)[:, 1].reshape(-1, 1)

# 2. CONSTRUIRE LA MATRICE DU MÉTA-MODÈLE (X_meta_train)
# On concatène les deux colonnes de probabilités côte à côte
X_meta_train = np.hstack((proba_train_log, proba_train_rf)) # Forme (891, 2)

# --- 3. CODAGE DU META-LEARNER (L'ARBRE DE DÉCISION FINAL) ---
# En C, tu aurais codé une structure de nœuds récursive. Ici, on appelle l'objet :
meta_tree = DecisionTreeClassifier(max_depth=3, random_state=42)

# L'arbre va apprendre les règles optimales sur les probabilités des deux modèles
meta_tree.fit(X_meta_train, y_train)

# --- 4. PHASE DE PRÉDICTION SÉQUENTIELLE SUR LE TEST ---
# Étape A : Obtenir les probabilités des modèles de base sur le jeu de test
proba_test_log = sigmoid(np.dot(X_test, W) + b) # Ton code NumPy normalisé
proba_test_rf = rf_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

# Étape B : Construire la matrice méta de test
X_meta_test = np.hstack((proba_test_log, proba_test_rf))

# Étape C : Prédiction finale par l'arbre méta
predictions_finales = meta_tree.predict(X_meta_test)

# Vos prédictions finales (0 ou 1) sont prêtes pour être envoyées sur Kaggle !

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# 1. Génération de données géométriques non-linéaires (Moons dataset)
X, y = make_moons(n_samples=300, noise=0.20, random_state=42)

# Toujours standardiser pour les modèles géométriques (KNN, SVM, Régression)
X = StandardScaler().fit_transform(X)

# 2. Définition des modèles du Blitz
modeles = {
    "Reg Logistique (Linéaire)": LogisticRegression(),
    "KNN (K=3, Géométrique local)": KNeighborsClassifier(n_neighbors=3),
    "SVM (Noyau Linéaire)": SVC(kernel="linear"),
    "SVM (Noyau RBF - Non-linéaire)": SVC(kernel="rbf", gamma=1)
}

# 3. Configuration de la grille de visualisation Matplotlib
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

# Création d'un maillage (mesh grid) pour tracer les frontières de décision
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 4. Entraînement et tracé des frontières
for idx, (nom, model) in enumerate(modeles.items()):
    ax = axes[idx]
    model.fit(X, y)
    
    # Prédiction sur chaque point du maillage pour dessiner la frontière
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Tracé des zones de décision
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    # Tracé des vrais points de données
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    ax.set_title(nom)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# --- MINI EXERCICE PCA EN BONUS ---
# Appliquons une ACP pour projeter nos données (2D) sur le composant principal (1D)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
print(f"Forme d'origine : {X.shape} | Forme après PCA : {X_pca.shape}")
print(f"Variance expliquée par le premier axe : {pca.explained_variance_ratio_[0]*100:.2f}%")

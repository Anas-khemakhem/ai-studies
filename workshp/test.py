import time
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

# 1. DONNÉES DU XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


# 2. MATHS DE BASE (Fonctions d'activation)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


# 3. INITIALISATION DU RÉSEAU (2 Entrées -> 4 Neurones cachés -> 1 Sortie)
np.random.seed(42)
W1 = np.random.uniform(-1, 1, (2, 4))
B1 = np.zeros((1, 4))
W2 = np.random.uniform(-1, 1, (4, 1))
B2 = np.zeros((1, 1))

lr = 0.5  # Taux d'apprentissage élevé pour que ça bouge vite à l'écran
losses = []
epochs_to_run = 2000

# Grille de points pour tracer la frontière de décision en arrière-plan
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 50), np.linspace(-0.5, 1.5, 50))
grid_points = np.c_[xx.ravel(), yy.ravel()]

print("Démarrage du laboratoire interactif...")
time.sleep(2)

for epoch in range(epochs_to_run + 1):
    # --- PHASE 1 : PASSE AVANT (Forward Pass) ---
    Z1 = np.dot(X, W1) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + B2
    A2 = sigmoid(Z2)

    # Calcul de l'erreur
    loss = np.mean((y - A2) ** 2)
    losses.append(loss)

    # --- PHASE 2 : RÉTROPROPAGATION (Backpropagation) ---
    error_out = A2 - y
    delta_out = error_out * sigmoid_derivative(Z2)

    error_hidden = np.dot(delta_out, W2.T)
    delta_hidden = error_hidden * sigmoid_derivative(Z1)

    # --- PHASE 3 : DESCENTE DE GRADIENT (Mise à jour) ---
    W2 -= lr * np.dot(A1.T, delta_out)
    B2 -= lr * np.sum(delta_out, axis=0, keepdims=True)
    W1 -= lr * np.dot(X.T, delta_hidden)
    B1 -= lr * np.sum(delta_hidden, axis=0, keepdims=True)

    # --- PHASE 4 : ANIMATION INTERACTIVE EN DIRECT ---
    # On rafraîchit l'affichage uniquement à des moments clés pour ne pas ralentir le CPU
    if epoch <= 10 or (epoch <= 100 and epoch % 10 == 0) or epoch % 100 == 0:
        clear_output(wait=True)

        # Calcul des prédictions sur toute la grille pour dessiner la frontière
        grid_Z1 = np.dot(grid_points, W1) + B1
        grid_A1 = sigmoid(grid_Z1)
        grid_Z2 = np.dot(grid_A1, W2) + B2
        grid_preds = sigmoid(grid_Z2).reshape(xx.shape)

        # Création de la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Graphique 1 : La frontière de décision non-linéaire
        contour = ax1.contourf(
            xx, yy, grid_preds, levels=20, cmap="RdYlBu", alpha=0.6
        )
        ax1.scatter(
            X[:, 0],
            X[:, 1],
            c=y.ravel(),
            cmap="RdYlBu",
            edgecolors="k",
            s=200,
            linewidths=2,
        )
        ax1.set_title(f"Frontière de Décision (Époque {epoch})")
        ax1.set_xlim(-0.2, 1.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Légende textuelle des prédictions actuelles du XOR
        for i, val in enumerate(X):
            ax1.text(
                val[0] + 0.05,
                val[1] + 0.05,
                f"Pred: {A2[i][0]:.2f}",
                fontsize=12,
                fontweight="bold",
                bbox=dict(
                    facecolor="white", alpha=0.8, edgecolor="none", pad=2
                ),
            )

        # Graphique 2 : Courbe d'erreur (Loss)
        ax2.plot(losses, color="crimson", linewidth=2)
        ax2.set_title(f"Courbe d'Erreur (Loss actuelle : {loss:.4f})")
        ax2.set_xlabel("Époques")
        ax2.set_ylabel("Erreur (MSE)")
        ax2.set_xlim(0, max(100, epoch))
        ax2.set_ylim(0, 0.3)
        ax2.grid(True)

        plt.tight_layout()
        display(fig)
        plt.close()

        # --- GESTION DU DÉBIT (Le fameux wait / sleep) ---
        if epoch < 5:
            time.sleep(2.5)  # Pause longue au tout début pour expliquer le mécanisme
        elif epoch <= 50:
            time.sleep(0.3)  # Ralentissement modéré pour voir la frontière commencer à se tordre
        else:
            time.sleep(0.01)  # Vitesse maximale pour la convergence finale

print("\nApprentissage réussi ! Le réseau a parfaitement résolu le XOR.")

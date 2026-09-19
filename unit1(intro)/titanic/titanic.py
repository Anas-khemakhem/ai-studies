import numpy as np
import pandas as pd

# Chargement des données
train_df = pd.read_csv("/ai studies/titanic/train.csv")
test_df = pd.read_csv("/ai studies/titanic/test.csv")

# Nettoyage des données (Traitement des valeurs manquantes pour Pclass également)
for df in [train_df, test_df]:
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age"] = df["Age"].fillna(train_df["Age"].median())
    df["Fare"] = df["Fare"].fillna(train_df["Fare"].median())
    df["Pclass"] = df["Pclass"].fillna(train_df["Pclass"].median())

# Sélection des 4 fonctionnalités
features = ["Pclass", "Sex", "Age", "Fare"]
X_train = train_df[features].values
y_train = train_df["Survived"].values.reshape(-1, 1)
X_test = test_df[features].values

# Initialisation des paramètres
W = np.zeros((len(features), 1))
b = 0.0
learning_rate = 0.0001
epochs = 1000
n = len(X_train)


# Fonction d'activation Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Boucle d'entraînement
for i in range(epochs):
    Z = np.dot(X_train, W) + b
    Y_pred = sigmoid(Z)
    loss = -np.mean(y_train * np.log(Y_pred + 1e-15) + (1 - y_train) * np.log(1 - Y_pred + 1e-15))
    dW = (1 / n) * np.dot(X_train.T, (Y_pred - y_train))
    db = (1 / n) * np.sum(Y_pred - y_train)
    W = W - (learning_rate * dW)
    b = b - (learning_rate * db)
    if i % 100 == 0:
        print(f"Étape {i} -> Erreur (Loss): {loss:.4f}")

print("\n--- Entraînement terminé ! ---")

# Prédictions sur le jeu de test
Z_test = np.dot(X_test, W) + b
probabilites_test = sigmoid(Z_test)
predictions = (probabilites_test >= 0.5).astype(int)
print(f"Forme des prédictions finales : {predictions.shape} (Prêt à être soumis !)")

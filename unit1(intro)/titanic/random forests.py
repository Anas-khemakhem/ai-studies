import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Chargement des données (Assure-toi d'avoir nettoyé les NaN au préalable !)
train_df = pd.read_csv("/ai studies/titanic/train.csv")
test_df = pd.read_csv("/ai studies/titanic/test.csv")

# Exemple de Features (Tu peux en ajouter d'autres !)
features = ['Pclass', 'Sex', 'Age', 'Fare']

# Conversion du Sexe en numérique pour Scikit-Learn
for df in [train_df, test_df]:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # Remplissage rapide des NaN restants pour le test
    df['Age'] = df['Age'].fillna(train_df['Age'].median())
    df['Fare'] = df['Fare'].fillna(train_df['Fare'].median())

X_train = train_df[features]
y_train = train_df['Survived']
X_test = test_df[features]

# 2. Initialisation de la Random Forest avec Hyperparamètres de contrôle
model = RandomForestClassifier(
    n_estimators=100,      # Nombre d'arbres dans la forêt
    max_depth=5,           # Limite la profondeur pour éviter le surapprentissage (Overfitting)
    random_state=42        # Assure la reproductibilité des résultats
)

# 3. Entraînement
model.fit(X_train, y_train)

# 4. Prédiction sur le jeu de test
predictions = model.predict(X_test)

# 5. Génération du fichier de soumission officiel pour Kaggle
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission_forest.csv", index=False)
print("Fichier submission_forest.csv créé avec succès !")

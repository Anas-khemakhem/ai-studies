import pandas as pd
import numpy as np

# 1. On crée le dictionnaire avec de vraies listes de données (CORRIGÉ)
data = {
    'PassengerId': [0,1,2,3,4],
    'Survived':[0,1,0,0,1],       # Notre cible Y (0=Mort, 1=Vivant)
    'Pclass':[1,2,3,2,1],         # Classe (1ère, 2ème, 3ème)
    'Age': [22.0, 38.0, 26.0, 35.0, np.nan], # np.nan = donnée manquante
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05]
}

# 2. On transforme ce dictionnaire en DataFrame Pandas
df = pd.DataFrame(data)

# --- LE RESTE DE TON CODE ---

# ÉTAPE A : Affiche les 3 premières lignes
print("--- Les 3 premières lignes ---")
print(df.head(3))

# ÉTAPE B : Observe les informations du tableau
print("\n--- Infos du DataFrame ---")
df.info()

# ÉTAPE C : Sépare les données pour l'IA
X = df[['Pclass', 'Fare']]
Y = df['Survived']

print("\n--- Vérification de X ---")
print(X)
print("\n--- Vérification de Y ---")
print(Y)
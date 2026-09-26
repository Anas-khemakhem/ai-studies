import torch
import torch.nn as nn
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=150)

# =====================================================================
# 🏗️ COUCHE DE CODAGE POSITIONNEL (Ton code complété)
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Étape D du challenge : addition sémantique de l'identité et du buffer de position
        x = x + self.pe[:, :x.size(1)]
        return x

# =====================================================================
# 🧪 APPLICATION : "Le chat mange la souris" VS "La souris mange le chat"
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # On définit une dimension d'embedding d_model = 4 (petite pour l'affichage de chiffres clairs)
    d_m = 4
    
    # 1. CRÉATION D'UN DICTIONNAIRE D'EMBEDDINGS FIXES POUR CHAQUE MOT
    # Dans un vrai modèle, ces vecteurs décrivent le sens sémantique du mot isolé.
    vocab_embeddings = {
        "le":     torch.tensor([ 0.5, -0.2,  0.1,  0.9]),
        "la":     torch.tensor([ 0.5, -0.2,  0.1,  0.8]), # "le" et "la" sont très proches sémantiquement
        "chat":   torch.tensor([ 2.1,  1.5, -0.8, -0.4]),
        "souris": torch.tensor([ 1.8,  0.9, -1.2, -0.1]), # "chat" et "souris" ont des profils de mammifères proches
        "mange":  torch.tensor([-0.5,  2.2,  3.0, -0.5])  # "mange" est un verbe (profil très différent)
    }

    # 2. CONSTRUCTION DES PHRASES SOUS FORME DE TENSEURS AVANT POSITION
    # Phrase 1 : ["le", "chat", "mange", "la", "souris"]
    p1_raw = torch.stack([vocab_embeddings[w] for w in ["le", "chat", "mange", "la", "souris"]]).unsqueeze(0)
    
    # Phrase 2 : ["la", "souris", "mange", "le", "chat"]
    p2_raw = torch.stack([vocab_embeddings[w] for w in ["la", "souris", "mange", "le", "chat"]]).unsqueeze(0)

    # 3. INSTANCIATION DE LA COUCHE POSITIONNELLE
    pe_layer = PositionalEncoding(d_model=d_m, max_len=10)
    
    # Application du codage positionnel
    p1_encoded = pe_layer(p1_raw)
    p2_encoded = pe_layer(p2_raw)

    print("===============================================================================")
    print("❌ ANALYSE AVANT COUCHE POSITIONNELLE : COMPARAISON DES MOTS IDENTIQUES")
    print("===============================================================================")
    print("Regarde le mot 'chat' aux deux positions différentes :")
    print("Phrase 1 - Index 1 ('chat' en position de sujet) :\n", p1_raw[0, 1].numpy())
    print("Phrase 2 - Index 4 ('chat' en position d'objet) :\n", p2_raw[0, 4].numpy())
    print("-> SÉMANTIQUE ISOLÉE PARFAITEMENT IDENTIQUE. L'IA NE SAIT PAS QUI MANGE QUI.")

    print("\n===============================================================================")
    print("🔥 ANALYSE APRÈS COUCHE POSITIONNELLE (ADDITION DES ONDES SIN/COS)")
    print("===============================================================================")
    print("Phrase 1 - Index 1 ('chat' + Signature de la Position 1) :\n", p1_encoded[0, 1].numpy())
    print("Phrase 2 - Index 4 ('chat' + Signature de la Position 4) :\n", p2_encoded[0, 4].numpy())
    print("\n-> LES MATRICES ONT CHANGÉ. Le mot 'chat' possède désormais deux coordonnées distinctes")
    print("dans l'espace mathématique selon son rôle syntaxique (Sujet vs Objet).")

    print("\n===============================================================================")
    print("📊 DIFFÉRENCE FINALE STRICTE DES DEUX TENSEURS DE PHRASES (Phrase 1 - Phrase 2)")
    print("===============================================================================")
    # Si le résultat était égal à 0, l'ordre des mots n'aurait aucun impact.
    # Ici, la matrice d'écart montre l'empreinte géométrique exacte du temps et de l'ordre.
    delta_phrases = (p1_encoded - p2_encoded).squeeze(0).numpy()
    print(delta_phrases)

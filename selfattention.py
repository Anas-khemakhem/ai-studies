import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        
        # Les 3 matrices de projection linéaire (représentées par des couches nn.Linear sans biais)
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        # X est la matrice d'entrée de taille (Batch_Size, Sequence_Length, d_model)
        # Pour cet exercice, on va extraire les dimensions en supposant un Batch_Size de 1
        # Forme de X : (Sequence_Length, d_model)
        if X.dim() == 3 and X.size(0) == 1:
            X = X.squeeze(0) # On simplifie la géométrie à (N, d_model)
            
        N, d_model = X.shape
        
        # 1. Calcul des projections Query, Key, Value
        Q = self.W_Q(X) # Taille (N, d_k)
        K = self.W_K(X) # Taille (N, d_k)
        V = self.W_V(X) # Taille (N, d_v)
        
        # --- CHALLENGE ALGÉBRIQUE ---
        # Étape A : Calcule les scores d'affinité bruts (Q multiplié par la transposée de K)
        # Note : La transposée de K de taille (N, d_k) s'obtient avec K.T ou K.transpose(0, 1)
        scores_bruts = torch.matmul(Q, K.T)
        
        # Étape B : Divise par le facteur d'échelle math.sqrt(self.d_k)
        scores_normalises = scores_bruts / math.sqrt(self.d_k)
        
        # Étape C : Passe le résultat dans la couche Softmax pour obtenir la matrice de poids d'attention
        poids_attention = self.softmax(scores_normalises)
        
        # Étape D : Multiplie les poids d'attention par la matrice des valeurs V
        sortie_contextuelle = torch.matmul(poids_attention, V)
        
        return sortie_contextuelle, poids_attention

# --- SCRIPT DE VÉRIFICATION DU FLUX QUANTIQUE ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution du moteur d'attention sur : {str(device).upper()}")
    
    # Imaginons une phrase de 4 mots ("Le", "Transformer", "est", "génial")
    seq_len = 4
    d_mod = 16   # Dimension d'embedding d'origine
    dk = 8       # Dimension de l'espace Query/Key
    dv = 8       # Dimension de l'espace Value
    
    # Matrice d'entrée aléatoire
    X_input = torch.randn(1, seq_len, d_mod, device=device)
    
    moteur_attention = ScaledDotProductAttention(d_model=d_mod, d_k=dk, d_v=dv).to(device)
    
    output, weights = moteur_attention(X_input)
    
    print("\n--- RAPPORT DU GRAPH DE CALCUL ---")
    print("Forme de la matrice de sortie (Doit être (4, 8)) :", output.shape)
    print("Forme de la matrice de poids d'attention (Doit être (4, 4)) :", weights.shape)
    print("\nVérification stochastique (la somme de chaque ligne des poids doit valoir 1.0) :\n", torch.sum(weights, dim=-1))

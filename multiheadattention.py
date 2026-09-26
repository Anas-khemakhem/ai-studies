import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

# Configuration de NumPy pour un affichage propre des matrices sans notation scientifique complexe
np.set_printoptions(precision=4, suppress=True, linewidth=150)

# =====================================================================
# 🏗️ ARCHITECTURE STANDARD DE MULTI-HEAD ATTENTION
# =====================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads !"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections globales unifiées (Poids initiaux aléatoires)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        
        Q = self.W_Q(q) 
        K = self.W_K(k)
        V = self.W_V(v)
        
        Q = self.split_heads(Q, batch_size) 
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        poids_attention = self.softmax(scores)
        context_layer = torch.matmul(poids_attention, V)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.d_model)
        
        return self.W_O(context_layer)

# =====================================================================
# 🔬 SCRIPT D'ENTRAÎNEMENT ET DE VISUALISATION DÉTAILLÉE DES POIDS
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42) # Fixation du hasard pour la reproductibilité
    
    # Dimensions réduites à dessein pour l'affichage de sous-matrices lisibles
    b_size = 1   # 1 seule phrase
    s_len = 3   # 3 mots
    d_m = 8     # d_model = 8 dimensions d'embedding
    n_heads = 2 # 2 têtes d'attention (Chaque tête travaille dans d_k = 4)
    
    # 1. Données d'entrée simulées
    X_input = torch.randn(b_size, s_len, d_m, requires_grad=True)
    
    # 2. Instanciation du modèle et configuration de l'apprentissage
    mha = MultiHeadAttention(d_model=d_m, num_heads=n_heads)
    
    # On ajoute une tête linéaire finale (notre classifieur) pour créer la Loss
    classifier_head = nn.Linear(d_m, 1, bias=False)
    
    # Utilisation du Gradient Descent pur (SGD) avec un gros taux d'apprentissage (lr=1.0)
    # pour forcer les poids à bouger violemment et rendre le changement flagrant à l'écran
    optimizer = optim.SGD(list(mha.parameters()) + list(classifier_head.parameters()), lr=1.0)
    criterion = nn.BCEWithLogitsLoss()
    
    # La cible idéale (Ground Truth)
    target = torch.tensor([[1.0]])

    # =================================================================
    # 📸 CAPTURE ET AFFICHAGE DES POIDS AVANT MISE À JOUR
    # =================================================================
    # On extrait une sous-section 3x3 en haut à gauche des matrices de poids réelles
    poids_Q_avant = mha.W_Q.weight.detach().numpy()[:3, :3].copy()
    poids_K_avant = mha.W_K.weight.detach().numpy()[:3, :3].copy()
    poids_V_avant = mha.W_V.weight.detach().numpy()[:3, :3].copy()
    poids_O_avant = mha.W_O.weight.detach().numpy()[:3, :3].copy()
    
    print("===============================================================================")
    print("🔴 ÉTAPE 0 : VRAIS POIDS DES MATRICES DE PROJECTION (AVANT APPRENTISSAGE - SOUS-MATRICES 3x3)")
    print("===============================================================================")
    print("Sous-matrice W_Q (Query) Initiale :\n", poids_Q_avant)
    print("\nSous-matrice W_K (Key) Initiale   :\n", poids_K_avant)
    print("\nSous-matrice W_V (Value) Initiale :\n", poids_V_avant)
    print("\nSous-matrice W_O (Output) Initiale :\n", poids_O_avant)

    # =================================================================
    # 🔄 ÉCOULEMENT ET CALCUL DES GRADIENTS (FORWARD + BACKWARD)
    # =================================================================
    # 1. Forward Pass
    mha_output = mha(X_input, X_input, X_input)
    cls_vector = mha_output[:, 0, :] # On prend le premier mot pour décider
    logit = classifier_head(cls_vector)
    loss = criterion(logit, target)
    
    # 2. Backward Pass
    optimizer.zero_grad()
    loss.backward()
    
    # 3. Mise à jour physique des poids dans la mémoire
    optimizer.step()

    # =================================================================
    # 📸 CAPTURE ET AFFICHAGE DES POIDS APRÈS MISE À JOUR
    # =================================================================
    poids_Q_apres = mha.W_Q.weight.detach().numpy()[:3, :3]
    poids_K_apres = mha.W_K.weight.detach().numpy()[:3, :3]
    poids_V_apres = mha.W_V.weight.detach().numpy()[:3, :3]
    poids_O_apres = mha.W_O.weight.detach().numpy()[:3, :3]
    
    print("\n===============================================================================")
    print("🟢 ÉTAPE 1 : VRAIS POIDS DES MATRICES DE PROJECTION (APRÈS MISE À JOUR - SOUS-MATRICES 3x3)")
    print("===============================================================================")
    print("Sous-matrice W_Q (Query) Finale :\n", poids_Q_apres)
    print("\nSous-matrice W_K (Key) Finale   :\n", poids_K_apres)
    print("\nSous-matrice W_V (Value) Finale :\n", poids_V_apres)
    print("\nSous-matrice W_O (Output) Finale :\n", poids_O_apres)

    print("\n===============================================================================")
    print("📊 ÉTAPE 2 : MATRICES DES VARIATIONS STRICTES (ΔW = APRÈS - AVANT)")
    print("===============================================================================")
    # Si le résultat contient des nombres non-nuls, la preuve est absolue : les poids ont changé !
    print("Variation réelle subie par W_Q (Delta) :\n", poids_Q_apres - poids_Q_avant)
    print("\nVariation réelle subie par W_K (Delta) :\n", poids_K_apres - poids_K_avant)
    print("\nVariation réelle subie par W_V (Delta) :\n", poids_V_apres - poids_V_avant)
    print("\nVariation réelle subie par W_O (Delta) :\n", poids_O_apres - poids_O_avant)
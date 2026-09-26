import torch
import torch.nn as nn

class CelluleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CelluleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Pour optimiser les calculs, on regroupe les matrices de poids des 4 portes 
        # en une seule grande couche linéaire de taille (input_dim + hidden_dim, 4 * hidden_dim)
        self.gates_layer = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(self, x, etats_precedents):
        # etats_precedents est un tuple (h_prev, c_prev)
        h_prev, c_prev = etats_precedents
        
        # Concaténation de l'état caché précédent et de l'entrée actuelle sur la dimension des features
        combined = torch.cat((h_prev, x), dim=1) # Forme : (Batch_Size, input_dim + hidden_dim)
        
        # Calcul simultané de toutes les portes
        all_gates = self.gates_layer(combined) # Forme : (Batch_Size, 4 * hidden_dim)
        
        # Découpage des blocs pour isoler chaque porte (Chunking)
        f_gate, i_gate, c_tilde, o_gate = torch.chunk(all_gates, 4, dim=1)
        
        # Application des fonctions d'activation non-linéaires
        f_gate = torch.sigmoid(f_gate)
        i_gate = torch.sigmoid(i_gate)
        c_tilde = torch.tanh(c_tilde)
        o_gate = torch.sigmoid(o_gate)
        
        # --- CHALLENGE INFORMATHÉMATIQUE ---
        # 1. Calcule le nouveau Cell State (c_next) en appliquant les portes f_gate et i_gate
        # Note : Utilise l'opérateur * pour la multiplication élément par élément (Hadamard)
        c_next = f_gate * c_prev + i_gate * c_tilde
        
        # 2. Calcule le nouveau Hidden State (h_next) à partir de o_gate et torch.tanh(c_next)
        h_next = o_gate * torch.tanh(c_next)
        
        return h_next, c_next

# --- SCRIPT DE VÉRIFICATION DU FLUX TEMPOREL ---
if __name__ == "__main__":
    batch_size = 3
    longueur_sequence = 5
    d_entree = 10
    d_cachee = 20
    
    # Simule un tenseur de texte vectorisé : (Batch, Temps, Features)
    sequence_input = torch.randn(batch_size, longueur_sequence, d_entree)
    
    lstm_custom = CelluleLSTM(input_dim=d_entree, hidden_dim=d_cachee)
    
    # Initialisation des états initiaux à zéro (h_0, c_0)
    h_t = torch.zeros(batch_size, d_cachee)
    c_t = torch.zeros(batch_size, d_cachee)
    
    print("--- SÉQUENÇAGE DU SENS ---")
    # Boucle sur la dimension temporelle de la séquence (Longueur = 5)
    for t in range(longueur_sequence):
        x_t = sequence_input[:, t, :] # Extraction du token à l'instant t -> (Batch, Features)
        
        # Rétroaction récursive
        h_t, c_t = lstm_custom(x_t, (h_t, c_t))
        print(f"Étape Temporelle {t+1} -> Forme de l'état caché h_t : {h_t.shape}")
        
    print("\nFlux validé sans erreur de tenseurs !")

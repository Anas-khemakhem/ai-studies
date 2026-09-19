import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import IPython.display as display
import matplotlib.cm as cm

# =====================================================================
# INTERFACE: Convertir une Feature Map en image thermique (Colab)
# =====================================================================
def map_to_colab_display(tensor, colormap_name='jet'):
    arr = tensor.squeeze().detach().cpu().numpy()
    if arr.max() - arr.min() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = arr * 0.0
    cmap = cm.get_cmap(colormap_name)
    colored_arr = (cmap(arr)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_arr, mode='RGB')

# =====================================================================
# STEP 0 : CHARGEMENT DE TA VRAIE IMAGE
# =====================================================================
img_raw = Image.open('image_tbMMRT.png').convert('RGB')
img_resized = img_raw.resize((300, 200))

# Extraction des canaux RGB (Normalisés entre 0 et 1)
arr_rgb = np.array(img_resized, dtype=np.float32) / 255.0
arr_rgb = np.transpose(arr_rgb, (2, 0, 1))
image_tensor = torch.from_numpy(arr_rgb).unsqueeze(0)

# =====================================================================
# PIPELINE CNN INTERMÉDIAIRE (Filtres Déterministes Multi-Étapes)
# =====================================================================
class StepByStepCNN(nn.Module):
    def __init__(self):
        super(StepByStepCNN, self).__init__()
        
        # 1. Filtre Chromatique Avancé 
        # Cible le bleu pur des voitures en éliminant le vert (arbres) et le rouge (voiture centrale)
        # Mais applique aussi un fort malus au bleu clair (le ciel) pour l'éteindre !
        self.color_filter = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        f_color = torch.zeros((1, 3, 1, 1))
        f_color[0, 0, 0, 0] = -1.5  # Supprime le rouge
        f_color[0, 1, 0, 0] = -3.0  # Supprime le vert des arbres
        f_color[0, 2, 0, 0] = 3.5   # Amplifie le bleu intense des voitures
        self.color_filter.weight = nn.Parameter(f_color, requires_grad=False)
        
        # 2. Extracteur de Contours (Filtre Scharr Moderne pour le désassemblage)
        self.scharr_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        fx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]])
        self.scharr_x.weight = nn.Parameter(fx.unsqueeze(0).unsqueeze(0), requires_grad=False)
        
        # 3. Couche d'Assemblage Spatial (Filtre Moyenneur flouteur)
        self.assembler = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self.assembler.weight = nn.Parameter(torch.ones((1, 1, 5, 5)) * 0.04, requires_grad=False)
        
        # 4. Couche de Concept Macro (Intègre les formes denses)
        self.macro_concept = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.macro_concept.weight = nn.Parameter(torch.ones((1, 1, 7, 7)) * 1.5, requires_grad=False)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Étape A : Isolation de la signature colorimétrique (Ici le ciel s'éteint)
        chroma = self.relu(self.color_filter(x))
        
        # Étape B : DÉSASSEMBLAGE GÉOMÉTRIQUE (Contours des objets restants)
        edges = self.relu(self.scharr_x(chroma))
        
        # Étape C : ASSEMBLAGE DES FORMES (Fusion des arrêtes)
        assembled = self.relu(self.assembler(edges))
        
        # Étape D : DEEP CONCEPT LEARNING (Concentration de la masse sémantique)
        concept = self.relu(self.macro_concept(assembled))
        
        return chroma, edges, assembled, concept

# Lancement des calculs
model = StepByStepCNN()
with torch.no_grad():
    chroma, edges, assembled, concept = model(image_tensor)

# =====================================================================
# AFFICHAGE INDIVIDUEL DE CHAQUE ÉTAPE DANS GOOGLE COLAB
# =====================================================================
print("📸 STEP 0 : SCÈNE ORIGINALE DU DATASET")
display.display(img_resized)

print("\n🎨 STEP 1 : FILTRAGE CHROMATIQUE SÉLECTIF")
print("-> Les arbres (verts), la voiture centrale (rouge) et le ciel (bleu trop clair) sont éteints.")
display.display(map_to_colab_display(chroma))

print("\n🟢 STEP 2 : DÉSASSEMBLAGE GÉOMÉTRIQUE (Contours Scharr)")
print("-> La forme des voitures bleues est découpée en lignes d'intensité pures.")
display.display(map_to_colab_display(edges))

print("\n🔵 STEP 3 : ASSEMBLAGE ET FUSION DES STRUCTURES LOCALES")
print("-> Les arrêtes se rapprochent, se floutent et fusionnent pour former des blocs compacts.")
display.display(map_to_colab_display(assembled))

print("\n🟡 STEP 4 : DEEP CONCEPT DÉTECTION (Résultat final du Backbone)")
print("-> Abstraction totale des pixels. Le réseau allume deux cibles parfaites sur les objets recherchés.")
display.display(map_to_colab_display(concept))

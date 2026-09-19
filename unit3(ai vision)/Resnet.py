import torch
import torch.nn as nn

#Lorsque tu fais une addition élément par élément F(x) + x, une contrainte informatique stricte s'applique : 
#la matrice de sortie de tes convolutions \(F(x)\) doit avoir exactement les mêmes dimensions (Canaux, Hauteur, Largeur) que ta matrice d'entrée \(x\).
#Si une couche convolutive utilise un stride=2, elle va diviser la hauteur et la largeur par 2. 
#Ton addition va crasher instantanément en mémoire. 
#Pour régler ce problème d'alignement, on applique une projection linéaire (une convolution 1 × 1) sur le chemin de la Skip Connection pour réaligner les dimensions de \(x\).

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Première convolution spatiale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # Normalisation de batch pour stabiliser l'apprentissage
        self.relu = nn.ReLU()
        
        # Seconde convolution spatiale (garde les mêmes dimensions)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Mécanisme d'alignement des dimensions pour la Skip Connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Si les dimensions ne collent pas, on applique une Conv 1x1 pour réaligner
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # On sauvegarde l'identité (l'entrée) en appliquant le raccourci d'alignement si nécessaire
        identite = self.shortcut(x)
        
        # Passage à travers le chemin convolutif lourd (F(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # CHALLENGE : Effectue l'addition résiduelle (F(x) + x) entre 'out' et 'identite'
        # Puis passe le résultat final dans l'activation finale self.relu()
        out = out + identite
        out = self.relu(out)
        
        return out

# --- VÉRIFICATION DIMENSIONNELLE DES TENSEURS ---
if __name__ == "__main__":
    # Cas 1 : Alignement parfait (stride=1, canaux identiques)
    # Simule un batch de 2 images RGB (3 canaux) de taille 32x32
    x1 = torch.randn(2, 3, 32, 32)
    bloc1 = ResidualBlock(in_channels=3, out_channels=3, stride=1)
    sortie1 = bloc1(x1)
    print("Cas 1 -> Forme d'entrée :", x1.shape, "| Forme de sortie :", sortie1.shape)
    
    # Cas 2 : Réduction de dimension spatiale (stride=2 et augmentation des canaux de 3 à 64)
    x2 = torch.randn(2, 3, 32, 32)
    bloc2 = ResidualBlock(in_channels=3, out_channels=64, stride=2)
    sortie2 = bloc2(x2)
    print("Cas 2 -> Forme d'entrée :", x2.shape, "| Forme de sortie (doit être (2, 64, 16, 16)) :", sortie2.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LaplacianOfGaussianCNN(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(LaplacianOfGaussianCNN, self).__init__()
        self.kernel_size = kernel_size
        
        # --- 1. CRÉATION DU FILTRE LoG MATHÉMATIQUE ---
        # On génère une grille de coordonnées (x, y) centrée en 0
        r = (kernel_size - 1) // 2
        x, y = np.mgrid[-r:r+1, -r:r+1]
        
        # Formule analytique exacte de la dérivée seconde de la Gaussienne (LoG)
        # Elle combine le lissage et la détection de contour en une seule équation
        norm_factor = 1.0 / (np.pi * (sigma ** 4))
        exponent = -(x**2 + y**2) / (2 * (sigma ** 2))
        laplacian_part = 1.0 - (x**2 + y**2) / (2 * (sigma ** 2))
        log_kernel = norm_factor * laplacian_part * np.exp(exponent)
        
        # Centrage parfait pour garantir que les zones plates de l'image donnent exactement 0
        log_kernel = log_kernel - np.mean(log_kernel)
        
        # --- 2. CONFIGURATION DE LA COUCHE PYTORCH ---
        # Version moderne : On utilise 1 seul out_channel pour une application isotrope (toutes directions)
        self.log_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, 
                                  padding=r, bias=False)
        
        # Conversion en Tenseur PyTorch de dimension (out_channels, in_channels, H, W) -> (1, 1, K, K)
        torch_kernel = torch.from_numpy(log_kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        # On injecte le filtre et on fige les gradients (requires_grad=False)
        self.log_conv.weight = nn.Parameter(torch_kernel, requires_grad=False)
        
    def forward(self, x):
        # 1. Extraction brute des passages par zéro via la convolution LoG
        log_response = self.log_conv(x)
        
        # 2. SEUILLAGE OPTIMISÉ POUR DES CONTOURS EXTRA NETS
        # Le LoG détecte le centre du contour là où le signal traverse 0.
        # On applique une valeur absolue pour obtenir des lignes blanches sur fond noir,
        # puis on seuille pour éliminer le bruit résiduel.
        contours = torch.abs(log_response)
        contours = torch.where(contours > 0.05, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
        
        return contours

# --- ZONE DE TEST SUR UN SIGNAL COMPLEXE ---
if __name__ == "__main__":
    # Simulation d'une image 8x8 contenant une forme complexe (un "L" inversé à forte intensité)
    image_test = torch.zeros((1, 1, 8, 8))
    image_test[0, 0, 2:6, 2:4] = 1.0  # Ligne verticale du L
    image_test[0, 0, 4:6, 4:6] = 1.0  # Ligne horizontale du L
    
    # Injection d'un bruit numérique aléatoire (pour tester la robustesse du lissage Gaussien)
    bruit = torch.randn((1, 1, 8, 8)) * 0.05
    image_bruitée = torch.clamp(image_test + bruit, 0.0, 1.0)
    
    print("--- 1. IMAGE ORIGINALE BRUITÉE ---")
    print(np.round(image_bruitée.numpy()[0, 0], 2))
    
    # Initialisation du modèle avec un noyau 5x5 stable
    model = LaplacianOfGaussianCNN(kernel_size=5, sigma=1.0)
    
    # Forward Pass ultra-rapide
    with torch.no_grad():
        output_contours = model(image_bruitée)
        
    print("\n--- 2. CONTOURS EXTRAITS VIA LOG VECTORISÉ (1.0 = Contour détecté) ---")
    print(output_contours.numpy()[0, 0])

import torch
import torch.nn as nn
import torchvision.ops as ops

class YOLODecoderAndLoss(nn.Module):
    def __init__(self, anchors):
        super(YOLODecoderAndLoss, self).__init__()
        # anchors: Tensor of shape (num_anchors, 2) -> [anchor_w, anchor_h]
        self.register_buffer('anchors', torch.tensor(anchors, dtype=torch.float32))
        self.num_anchors = self.anchors.shape[0]

    def decode_predictions(self, output_tensor):
        """
        Transforme le tenseur brut du réseau en coordonnées de boîte réelles.
        Input: output_tensor de forme (Batch, Anchors * (4 + 1 + Classes), H, W)
        """
        B, C, H, W = output_tensor.shape
        K = (C // self.num_anchors) - 5 # Nombre de classes
        
        # Reshape pour isoler les prédictions par Ancre: (B, Anchors, H, W, 5 + K)
        predictions = output_tensor.view(B, self.num_anchors, 5 + K, H, W).permute(0, 1, 3, 4, 2).contiguous()
        
        # Scission des canaux du tenseur
        t_x = predictions[..., 0]
        t_y = predictions[..., 1]
        t_w = predictions[..., 2]
        t_h = predictions[..., 3]
        objectness = torch.sigmoid(predictions[..., 4])
        class_probs = torch.sigmoid(predictions[..., 5:])
        
        # Génération de la grille de coordonnées (c_x, c_y)
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_x = grid_x.to(output_tensor.device).view(1, 1, H, W)
        grid_y = grid_y.to(output_tensor.device).view(1, 1, H, W)
        
        # Récupération des dimensions des ancres alignées sur la forme du tenseur
        p_w = self.anchors[:, 0].view(1, self.num_anchors, 1, 1)
        p_h = self.anchors[:, 1].view(1, self.num_anchors, 1, 1)
        
        # APPLICATION DES FORMULES MATHÉMATIQUES DE DÉCODAGE
        b_x = torch.sigmoid(t_x) + grid_x
        b_y = torch.sigmoid(t_y) + grid_y
        b_w = p_w * torch.exp(t_w)
        b_h = p_h * torch.exp(t_h)
        
        # Conversion au format standard (x_min, y_min, x_max, y_max) normalisé par la taille de la grille
        x1 = (b_x - b_w / 2) / W
        y1 = (b_y - b_h / 2) / H
        x2 = (b_x + b_w / 2) / W
        y2 = (b_y + b_h / 2) / H
        
        decoded_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return decoded_boxes, objectness, class_probs

    def compute_ciou_loss(self, pred_boxes, target_boxes):
        """
        Calcule la perte CIoU matricielle entre les boîtes décodées et la Ground Truth.
        Forme des inputs : (N, 4) où le format est [x1, y1, x2, y2]
        """
        # Utilisation de l'opérateur optimisé natif de torchvision pour le CIoU loss
        # Il implémente la formule complète incluant l'IoU, la distance des centres et le ratio d'aspect.
        ciou = ops.complete_box_iou_loss(pred_boxes, target_boxes, reduction='mean')
        return ciou

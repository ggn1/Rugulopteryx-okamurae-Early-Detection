# IMPORTS
import torch

def iou_score(preds, targets, threshold=0.5, eps=1e-7):
    """
    preds: torch tensor logits or probabilities (B,1,H,W)
    targets: tensor (B,1,H,W)
    """
    if preds.dtype != torch.uint8 and preds.dtype != torch.bool:
        probs = torch.sigmoid(preds)
        preds_bin = (probs > threshold).float()
    else:
        preds_bin = preds.float()
    inter = (preds_bin * targets).sum(dim=(1,2,3))
    union = ((preds_bin + targets) >= 1).float().sum(dim=(1,2,3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()
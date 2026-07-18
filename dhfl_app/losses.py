"""Loss functions and evaluation metrics."""
from typing import Optional, Protocol
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


class SegmentationModel(Protocol):
    def __call__(self, x: Tensor, error_map: Optional[Tensor] = None,
                 return_debug: bool = False): ...


def _check_binary_tensors(logits: Tensor, targets: Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(f"shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"expected [B, 1, H, W], got {tuple(logits.shape)}")


def soft_dice_loss(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    _check_binary_tensors(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    return (1.0 - ((2.0 * inter + eps) / (denom + eps))).mean()


def bce_dice_loss(
    logits: Tensor,
    targets: Tensor,
    pixel_weight: Optional[Tensor] = None,
    pos_weight: float = 2.0,
) -> Tensor:
    _check_binary_tensors(logits, targets)
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    if pixel_weight is not None:
        bce = (bce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
    else:
        bce = bce.mean()
    return bce + soft_dice_loss(logits, targets)


@torch.no_grad()
def binary_iou_score(logits: Tensor, targets: Tensor,
                      threshold: float = 0.5, eps: float = 1e-6) -> Tensor:
    _check_binary_tensors(logits, targets)
    preds = (torch.sigmoid(logits) >= threshold).to(targets.dtype)
    dims = (1, 2, 3)
    inter = torch.sum(preds * targets, dim=dims)
    union = torch.sum((preds + targets) > 0, dim=dims).to(targets.dtype)
    return ((inter + eps) / (union + eps)).mean()


@torch.no_grad()
def evaluate_iou(
    model: SegmentationModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    scores = []
    for i, (image, mask) in enumerate(loader):
        if i >= max_batches:
            break
        logits = model(image.to(device))
        scores.append(binary_iou_score(logits, mask.to(device)))
    return float(torch.stack(scores).mean().cpu())


def error_map_from_prediction(pred_logits: Tensor, corrected_mask: Tensor) -> Tensor:
    return corrected_mask.float() - torch.sigmoid(pred_logits).detach()


def error_weight_map(error_map: Tensor, base: float = 1.0, gain: float = 4.0) -> Tensor:
    return base + gain * error_map.abs().detach()
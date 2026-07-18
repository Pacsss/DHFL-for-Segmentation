from typing import Dict
import torch
from torch import Tensor


def create_anchor(model) -> Dict[str, Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.adapter.named_parameters()
    }


def anchor_regularization_loss(model, anchor: Dict[str, Tensor],
                                anchor_strength: float) -> Tensor:
    """L_anchor = anchor_strength * sum(||w - w_anchor||^2)"""
    device = next(model.adapter.parameters()).device
    loss = torch.tensor(0.0, device=device)
    for name, param in model.adapter.named_parameters():
        if name in anchor:
            loss = loss + (param - anchor[name]).pow(2).sum()
    return anchor_strength * loss


def maybe_update_anchor(
    anchor: Dict[str, Tensor],
    model,
    current_iou: float,
    best_iou: float,
    threshold: float = 0.001,
) -> float:
    if current_iou - best_iou > threshold:
        anchor.update({
            name: param.detach().clone()
            for name, param in model.adapter.named_parameters()
        })
        return current_iou
    return best_iou
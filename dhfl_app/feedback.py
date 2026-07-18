"""One human-feedback training step, with gradient-to-zero and anchor control."""
from dataclasses import dataclass
import torch
from torch import Tensor

from .losses import bce_dice_loss, binary_iou_score, error_map_from_prediction, error_weight_map
from .anchor import anchor_regularization_loss


@dataclass(frozen=True)
class FeedbackStepResult:
    loss: float
    iou_before: float
    iou_after: float
    mean_abs_error: float
    delta_abs_before: float
    delta_abs_after: float


def feedback_step(
    model,
    optimizer: torch.optim.Optimizer,
    image: Tensor,
    corrected_mask: Tensor,
    anchor: dict,
    error_gain: float = 4.0,
    use_error_weights: bool = True,
    anchor_strength: float = 1e-2,
    correction_denom: float = 0.3,
    max_grad_norm: float = 1.0,
    
) -> FeedbackStepResult:
    model.train()
    corrected_mask = corrected_mask.float()

    with torch.no_grad():
        before = model(image, return_debug=True)
        error_map = error_map_from_prediction(before.logits, corrected_mask)
        iou_before = binary_iou_score(before.logits, corrected_mask)
        delta_abs_before = before.delta_features.abs().mean()
        mean_abs_error = error_map.abs().mean()
        correction_signal = mean_abs_error / (mean_abs_error + correction_denom)

    pixel_weight = error_weight_map(error_map, gain=error_gain) if use_error_weights else None

    optimizer.zero_grad(set_to_none=True)
    out = model(image, error_map=error_map, return_debug=True)
    seg_loss = bce_dice_loss(out.logits, corrected_mask, pixel_weight=pixel_weight)
    delta_reg = 1e-4 * out.delta_features.pow(2).mean()
    correction_loss = (seg_loss + delta_reg) * correction_signal
    anc_loss = anchor_regularization_loss(model, anchor, anchor_strength)
    loss = correction_loss + anc_loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), max_norm=max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        after = model(image, return_debug=True)
        iou_after = binary_iou_score(after.logits, corrected_mask)
        delta_abs_after = after.delta_features.abs().mean()

    return FeedbackStepResult(
        loss=float(loss.detach().cpu()),
        iou_before=float(iou_before.cpu()),
        iou_after=float(iou_after.cpu()),
        mean_abs_error=float(mean_abs_error.cpu()),
        delta_abs_before=float(delta_abs_before.cpu()),
        delta_abs_after=float(delta_abs_after.cpu()),
    )
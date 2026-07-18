"""
Deep Human Feedback Learning (DHFL)

Public API exports.
"""

from .adapter import ErrorAwareResidualAdapter

from .feedback import (
    feedback_step,
    FeedbackStepResult,
)

from .losses import (
    soft_dice_loss,
    bce_dice_loss,
    binary_iou_score,
    evaluate_iou,
    error_map_from_prediction,
    error_weight_map,
)

from .anchor import (
    create_anchor,
    anchor_regularization_loss,
    maybe_update_anchor,
)

__all__ = [
    # Adapter
    "ErrorAwareResidualAdapter",

    # Feedback
    "feedback_step",
    "FeedbackStepResult",

    # Losses / Metrics
    "soft_dice_loss",
    "bce_dice_loss",
    "binary_iou_score",
    "evaluate_iou",
    "error_map_from_prediction",
    "error_weight_map",

    # Anchor
    "create_anchor",
    "anchor_regularization_loss",
    "maybe_update_anchor",
]

__version__ = "1.0.0"
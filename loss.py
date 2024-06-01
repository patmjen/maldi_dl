from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCELoss(nn.Module):
    def __init__(
        self,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None,
        label_smoothing: Optional[float] = 0.0,
    ):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f'Uknown reduction: {reduction}')
        self.reduction = reduction
        self.weight = weight
        self.label_smoothing = label_smoothing


    def forward(self, input, target, mask):
        """
        Args:
            pred (torch.Tensor): (N, C, *) Predicted logits.
            label (torch.Tensor): (N, *) Label.
            mask (torch.Tensor): (N, *) Binary mask.

        Returns:
            torch.Tensor: Loss.
        """
        # Compute cross entropy loss
        loss = F.cross_entropy(
            input,
            target,
            reduction='none',
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )

        # Mask out loss for voxels outside the mask
        loss = loss * mask.to(loss.dtype)

        # Reduce loss
        if self.reduction == 'mean':
            loss_sum = loss.sum()
            mask_sum = mask.sum()
            if mask_sum == 0.0:
                # If the mask is empty, there is no meaningful gradient
                # Thus, we can just return the loss, which will be 0
                loss = loss_sum
            else:
                loss = loss_sum / mask_sum
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha


    def forward(self, input, target):
        return focal_loss(input, target, self.gamma, self.alpha).mean()


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
    s_j is the unnormalized score for class j.
    """
    # From: https://docs.monai.io/en/stable/_modules/monai/losses/focal_loss.html#FocalLoss.forward
    if target.ndim == 1:
        # Convert to one-hot
        target = F.one_hot(target, input.shape[1])
    input_ls = input.log_softmax(1)
    loss: torch.Tensor = -(1 - input_ls.exp()).pow(gamma) * input_ls * target

    if alpha is not None:
        # (1-alpha) for the background class and alpha for the other classes
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss

    return loss
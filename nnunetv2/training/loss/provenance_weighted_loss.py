#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Provenance-weighted Dice + cross-entropy loss, by Remi Hattat.
#
# The target's second channel, if present, holds a per-voxel weight in
# [0, 1], encoded as an integer 0-WEIGHT_SCALE so it survives the same
# on-disk dtype as the label. Both the CE and Dice terms use this weight
# map; a voxel weighted at 0 contributes to neither term. With a
# single-channel target (no weights injected), the loss falls back to
# plain unweighted Dice + CE.

import torch
from torch import nn

from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

WEIGHT_SCALE = 1000  # must match inject_weights.WEIGHT_SCALE


class WeightedRobustCrossEntropyLoss(nn.Module):
    """
    CrossEntropy that transparently handles both:
      - target with 1 channel  -> plain mean-reduced CE (ce_standard)
      - target with 2 channels -> per-voxel CE, weighted-averaged by the
        provenance weight map (channel 1, decoded from int16 / WEIGHT_SCALE)
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce_standard = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] == 1:
            labels = target[:, 0].long()
            return self.ce_standard(net_output, labels)

        labels = target[:, 0].long()
        weight = (target[:, 1].float() / WEIGHT_SCALE).clamp(0.0, 1.0)
        ce_per_voxel = self.ce(net_output, labels)
        return (ce_per_voxel * weight).sum() / weight.sum().clamp(min=1e-8)


class DC_and_WeightedCE_loss(nn.Module):
    def __init__(self, dice_kwargs: dict, ce_kwargs: dict, weight_dice: float = 1.0, weight_ce: float = 1.0):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **dice_kwargs)
        self.ce = WeightedRobustCrossEntropyLoss(**ce_kwargs)

        self._detected = False
        self._has_weights = False

    def _maybe_log_detection(self, target: torch.Tensor):
        if self._detected:
            return
        self._detected = True
        self._has_weights = target.shape[1] == 2
        if self._has_weights:
            w = (target[:, 1].float() / WEIGHT_SCALE).clamp(0.0, 1.0)
            print(f"provenance weights detected: target shape={tuple(target.shape)}, "
                  f"channels={target.shape[1]}. mean_w={w.mean().item():.4f}, "
                  f"min={w.min().item():.4f}, max={w.max().item():.4f}, "
                  f"pct<1.0: {(w < 1.0).float().mean().item() * 100:.1f}%")
        else:
            print(f"no weight channel (target shape={tuple(target.shape)}). standard mode.")

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._maybe_log_detection(target)

        labels = target[:, 0:1].long()
        loss_mask = None
        if target.shape[1] == 2:
            loss_mask = (target[:, 1:2].float() / WEIGHT_SCALE).clamp(0.0, 1.0)

        dc_loss = self.dc(net_output, labels, loss_mask=loss_mask)
        ce_loss = self.ce(net_output, target)

        return self.weight_dice * dc_loss + self.weight_ce * ce_loss

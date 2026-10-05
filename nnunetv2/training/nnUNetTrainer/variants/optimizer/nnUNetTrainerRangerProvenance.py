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
# Provenance-weighted training variant, by Remi Hattat.
#
# The target tensor can carry two channels instead of one: channel 0 is the
# label, channel 1 is a per-voxel provenance weight (see
# provenance_weighted_loss.py). The loss module detects this at the first
# training batch and switches between plain and weighted mode automatically.

import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerRanger import nnUNetTrainerRanger
from nnunetv2.training.loss.provenance_weighted_loss import DC_and_WeightedCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import SoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
import numpy as np


class nnUNetTrainerRangerProvenance(nnUNetTrainerRanger):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.print_to_log_file(
            'nnUNetTrainerRangerProvenance initialized. Weight detection at first batch.'
        )

    @property
    def has_provenance_weights(self):
        # exposed on the trainer since debug.json and logging read trainer attributes directly
        loss = self.loss.loss if hasattr(self.loss, 'loss') else self.loss
        return getattr(loss, '_has_weights', False)

    @property
    def _weight_detection_done(self):
        loss = self.loss.loss if hasattr(self.loss, 'loss') else self.loss
        return getattr(loss, '_detected', False)

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerRangerProvenance only supports standard multiclass " \
            "segmentation (has_regions=False); region-based (multi-label/BCE) " \
            "targets are not handled here."

        loss = DC_and_WeightedCE_loss(
            dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 'do_bg': False,
                         'smooth': 0, 'ddp': self.is_ddp},
            ce_kwargs={'ignore_index': self.label_manager.ignore_label
                       if self.label_manager.has_ignore_label else -100},
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def validation_step(self, batch: dict) -> dict:
        """
        Same as nnUNetTrainer.validation_step, except the online tp/fp/fn
        metrics are computed against the label channel only. Without
        stripping the weight channel first, get_tp_fp_fn_tn() scatters the
        raw 2-channel target as if it were class indices, which crashes on
        GPU as soon as the weight channel holds a value >= num_classes
        (weights are encoded up to 1000, so this happens almost
        immediately). The loss itself still gets the full 2-channel target,
        since it needs the weight channel.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)
            if hasattr(self.loss, 'compound_loss'):
                comp_l = self.loss.compound_loss
            else:
                comp_l = {}

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # strip the weight channel (if present) before computing online metrics
        if target.shape[1] == 2:
            target = target[:, 0:1]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        out = {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out


class nnUNetTrainerRangerProvenance_10epochs(nnUNetTrainerRangerProvenance):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10


class nnUNetTrainerRangerProvenance_100epochs(nnUNetTrainerRangerProvenance):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100


class nnUNetTrainerRangerProvenance_250epochs(nnUNetTrainerRangerProvenance):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250

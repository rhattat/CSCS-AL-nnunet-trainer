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
# Modifications by Remi Hattat: Ranger22 optimizer + scheduler wiring
# (nnUNetTrainerRanger and its epoch-count variants below).

import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.training.lr_scheduler.ranger22_scheduler import Ranger22Scheduler
from nnunetv2.training.optimizer.ranger22_optimizer import Ranger22
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerRanger(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = Ranger22(list(self.network.parameters()) + list(self.loss.parameters()), self.initial_lr,
                              weight_decay=self.weight_decay, amsgrad=True)
        lr_scheduler = Ranger22Scheduler(optimizer, self.num_iterations_per_epoch, self.num_epochs)
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # autocast only kicks in on cuda: it's slow on cpu and unsupported on mps
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)
            if hasattr(self.loss, 'compound_loss'):
                comp_l = self.loss.compound_loss
            else:
                comp_l = {}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
        else:
            l.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        out = {'loss': l.detach().cpu().numpy()}
        out.update({'comp_loss_%s' % l_i: comp_l[l_i].detach().cpu().numpy() for l_i in comp_l})
        return out

    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        if hasattr(self.loss.loss, 'update_weights') and callable(self.loss.loss.update_weights):
            self.loss.loss.update_weights(n_epoch=self.current_epoch, total_epoch=self.num_epochs)


class nnUNetTrainerRanger_noSmooth_10epochs(nnUNetTrainerRanger, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10


class nnUNetTrainerRanger_noSmooth_100epochs(nnUNetTrainerRanger, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100


class nnUNetTrainerRanger_noSmooth_250epochs(nnUNetTrainerRanger, nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250

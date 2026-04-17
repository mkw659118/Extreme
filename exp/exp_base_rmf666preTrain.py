import copy
import os
import time

import torch
from tqdm import trange

from exp.exp_base_rmf666 import BasicModel as BasicModelRMF5


class BasicModel(BasicModelRMF5):
    def __init__(self, config):
        super().__init__(config)
        self.pretrain_scaler = torch.amp.GradScaler(config.device)

    def setup_pretrain_optimizer(self, config):
        self.to(config.device)
        pretrain_lr = getattr(config, 'pretrain_lr', config.lr)
        pretrain_decay = getattr(config, 'pretrain_decay', config.decay)
        self.pretrain_optimizer = torch.optim.AdamW(
            self.model.get_state_prior_parameters(),
            lr=pretrain_lr,
            weight_decay=pretrain_decay,
        )
        self.pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.pretrain_optimizer,
            mode='min',
            factor=0.5,
            patience=max(1, config.patience // 5),
            threshold=1e-4,
        )

    def pretrain_one_epoch(self, dataModule):
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()

        total_loss = 0.0
        sample_count = 0
        last_aux = None

        for batch_idx, train_batch in enumerate(dataModule.train_data_loader):
            x, _, _, _= train_batch
            x = x.to(self.config.device, non_blocking=True)
            self.pretrain_optimizer.zero_grad(set_to_none=True)

            if self.config.use_amp:
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss, aux = self.model.pretrain_state_prior_loss(x)
                self.pretrain_scaler.scale(loss).backward()
                self.pretrain_scaler.unscale_(self.pretrain_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.get_state_prior_parameters(), max_norm=1.0)
                self.pretrain_scaler.step(self.pretrain_optimizer)
                self.pretrain_scaler.update()
            else:
                loss, aux = self.model.pretrain_state_prior_loss(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.get_state_prior_parameters(), max_norm=1.0)
                self.pretrain_optimizer.step()

            total_loss += loss.item() * x.size(0)
            sample_count += x.size(0)
            last_aux = aux

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataModule.train_data_loader):
                q_mean = aux['q_mean']
                print(
                    f"[pretrain] Batch {batch_idx + 1}/{len(dataModule.train_data_loader)} - "
                    f"Loss: {loss.item():.6f}, "
                    f"NLL: {aux['pretrain_nll'].item():.6f}, "
                    f"KL: {aux['balance_kl'].item():.4f}, "
                    f"Dom: {aux['dominant_penalty'].item():.4f}, "
                    f"q_mean={['%.3f' % v for v in q_mean.tolist()]}"
                )

        self.eval()
        torch.set_grad_enabled(False)
        t2 = time.time()
        avg_loss = total_loss / max(sample_count, 1)
        print(f"[Train-pretrain] Epoch {self.current_epoch} finished | Avg NLL: {avg_loss:.6f} | Time Cost: {t2 - t1:.2f}s")
        return avg_loss, t2 - t1, last_aux

    def _need_retrain(self, config, runId, log):
        model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'
        return (
            config.retrain == 1
            or (not os.path.exists(model_path) and config.continue_train)
        )

    def RunOnce(self, config, runId, model, datamodule, log):
        pretrain_epochs = int(getattr(config, 'pretrain_epochs', 10))
        freeze_after_pretrain = bool(getattr(config, 'freeze_prior_after_pretrain', False))

        if self._need_retrain(config, runId, log) and pretrain_epochs > 0:
            print('*******State Prior Pretraining*******')
            self.setup_pretrain_optimizer(config)
            best_loss = float('inf')
            best_state = copy.deepcopy(self.model.state_dict())

            for epoch in trange(pretrain_epochs, desc='State Prior Pretraining'):
                self.current_epoch = epoch + 1

                t_start = float(getattr(config, 'state_prior_temperature_start', getattr(config, 'state_prior_temperature', 1.0)))
                t_end = float(getattr(config, 'state_prior_temperature_end', 0.6))
                if pretrain_epochs > 1:
                    ratio = epoch / float(pretrain_epochs - 1)
                else:
                    ratio = 1.0
                self.model.state_prior.temperature = t_start + (t_end - t_start) * ratio

                train_nll, _, last_aux = self.pretrain_one_epoch(datamodule)
                self.pretrain_scheduler.step(train_nll)
                print(
                    f"[Pretrain] Current LR: {self.pretrain_optimizer.param_groups[0]['lr']:.6g}, "
                    f"Temp: {self.model.state_prior.temperature:.4f}"
                )
                if train_nll < best_loss:
                    best_loss = train_nll
                    best_state = copy.deepcopy(self.model.state_dict())

            self.model.load_state_dict(best_state)
            qmax = 0.0
            collapse_threshold = min(0.95, float(getattr(config, 'state_dom_cap', 0.8)) + 0.1)
            if 'last_aux' in locals() and last_aux is not None and 'q_mean' in last_aux:
                qmax = float(last_aux['q_mean'].max().item())

            if freeze_after_pretrain and qmax < collapse_threshold:
                self.model.freeze_state_prior()
                print('[Pretrain] state prior is frozen for backbone training')
            else:
                self.model.unfreeze_state_prior()
                print(f'[Pretrain] keep state prior trainable in backbone stage (qmax={qmax:.3f})')

        return super().RunOnce(config, runId, model, datamodule, log)

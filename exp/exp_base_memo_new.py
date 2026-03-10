import torch
import time
import contextlib
from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer
import os
from tqdm import *
import pickle
from utils.model_monitor import EarlyStopping

class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.pred_len = config.pred_len
        self.label_len = config.label_len
        device_str = str(config.device)
        self.device_type = 'cuda' if 'cuda' in device_str else 'cpu'
        self.scaler = torch.amp.GradScaler(config.device)
        self.current_epoch = 0  # 从0开始计数，第1个epoch训练时会更新为1

    def forward(self, *x, **kwargs):
        y = self.model(*x, **kwargs)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer = get_optimizer(self.parameters(), lr=config.lr, decay=config.decay, config=config)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.patience // 5,
            threshold=1e-3
        )

    def RunOnce(self, config, runId, model, datamodule, log):
        try:
            # 一些模型（如Keras兼容模型）可能需要compile，跳过非必要的compile
            model.compile()
        except Exception as e:
            print(f'Skip the model.compile() because {e}')

        # 设置EarlyStopping监控器
        monitor = EarlyStopping(config)

        # 创建保存模型的目录
        os.makedirs(f'./checkpoints/{config.model}', exist_ok=True)
        model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'

        # 判断是否需要重新训练：
        # 若 config.retrain==1 表示强制重训；
        # 或者模型文件不存在 且 设置了 continue_train，则需要重新训练
        retrain_required = config.retrain == 1 or not os.path.exists(model_path) and config.continue_train

        # 如果无需重新训练且已有模型文件，则直接加载模型并评估测试集性能
        if not retrain_required:
            try:
                # 加载之前记录的训练时间
                sum_time = pickle.load(open(f'./results/metrics/' + log.filename + '.pkl', 'rb'))['train_time'][runId]
                # 加载模型权重（weights_only=True 可忽略 optimizer 等无关信息）
                model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
                model.setup_optimizer(config)  # 重新设置优化器
                results = model.test(datamodule)  # 在测试集评估性能
                log.show_results(results, sum_time)
                config.record = False  # 不再记录当前结果
            except Exception as e:
                log.only_print(f'Error: {str(e)}')
                retrain_required = True  # 若加载失败则触发重新训练

        # 若设置为继续训练（即接着上次的结果继续）
        if config.continue_train:
            log.only_print(f'Continue training...')
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))

        # 若需要重新训练
        if retrain_required:
            model.setup_optimizer(config)
            train_time = []
            for epoch in trange(config.epochs):
                model.current_epoch = epoch + 1  # 从1开始计数，更符合实际习惯

                if monitor.early_stop:
                    break  # 若满足early stopping条件则提前终止训练

                # 训练一个epoch并记录耗时
                train_loss, time_cost = model.train_one_epoch(datamodule)
                train_time.append(time_cost)

                # 验证集上评估当前模型误差
                valid_error = model.valid(datamodule)

                model.scheduler.step(valid_error[config.monitor_metric])
                print(f"Current LR: {model.optimizer.param_groups[0]['lr']:.6g}")

                # 将当前epoch的验证误差传递给early stopping模块进行跟踪
                monitor.track_one_epoch(epoch, model, valid_error, config.monitor_metric)

                # 输出当前epoch的训练误差和验证误差，并记录训练时间
                log.show_epoch_error(runId, epoch, monitor, train_loss, valid_error, train_time)

                # 更新日志可视化（如绘图）
                log.plotter.append_epochs(train_loss, valid_error)

                # 暂存模型参数（即使不是最优，也为了中断续训做准备）
                torch.save(model.state_dict(), model_path)

            # 加载最优模型参数（来自early stopping）
            model.load_state_dict(monitor.best_model)

            # 建立历史库索引
            train_data = datamodule.train_data_loader.dataset
            train_loader = datamodule.train_data_loader
            model.prepare_retrieval_index(train_data, train_loader)

            # 累计训练时间（仅使用前best_epoch轮）
            sum_time = sum(train_time[: monitor.best_epoch])

            # 使用最优模型在测试集评估
            results = model.test(datamodule)
        
            log.show_test_error(runId, monitor, results, sum_time)

            # 保存最优模型参数
            torch.save(monitor.best_model, model_path)
            log(f'Model parameters saved to {model_path}')

        # 将训练时间加入返回结果中
        results['train_time'] = sum_time
        
        return results

    
    # 将差分域还原回原始域
    def _restore_to_raw(self, pred, label, mean, std):
        mean_t = torch.as_tensor(mean, device=pred.device, dtype=pred.dtype)
        std_t  = torch.as_tensor(std,  device=pred.device, dtype=pred.dtype)
        pred_diff_z = pred[:, :, 0]
        pred_diff   = pred_diff_z * std_t + mean_t
        y_pre       = label[:, 0, 3]
        pred_raw    = y_pre.unsqueeze(1) + torch.cumsum(pred_diff, dim=1)
        real_raw    = label[:, :, -1]
        return pred_raw, real_raw

    def prepare_retrieval_index(self, train_data, train_loader):
        print('*******Constructing the Retrieval Indexes*********')
        time_now = time.time()
        train_steps = len(train_loader)
        self.model.construct_index(len(train_data))
        with torch.no_grad():
            for epoch in range(1):
                iter_count = 0
                for i, (batch_x, x_mark, batch_y, sample_ids) in enumerate(train_loader):
                    batch_x = batch_x.float().to(self.config.device)
                    batch_x = batch_x.float().to(self.config.device)
                    batch_y = batch_y.float().to(self.config.device)
                    sample_ids = sample_ids.to(self.config.device)
                    iter_count += 1
                    self.model.add_key_value(batch_x[:,:, 0:1], batch_y[:, -self.config.pred_len:, 0:1], sample_ids)

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1}".format(i + 1, epoch + 1))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((1 - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

        print('*******Finishing the Retrieval Indexes*********')
        self.model.value_permute = self.model.values.permute(2,0,1)
        
    def train_one_epoch(self, dataModule):

        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()

        total_loss = 0.0
        sample_count = 0

        scaler = self.scaler

        try:
            for batch_idx, train_batch in enumerate(dataModule.train_data_loader):
                x, x_mark, label, sample_ids = train_batch

                x = x.to(self.config.device, non_blocking=True)
                x_mark = x_mark.to(self.config.device, non_blocking=True)
                label = label.to(self.config.device, non_blocking=True)
                sample_ids = sample_ids.to(self.config.device, non_blocking=True).long()

                # decoder input
                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat([label[:, :self.label_len, :], dec_input], dim=1).float().to(self.config.device)

                self.optimizer.zero_grad(set_to_none=True)

                if self.config.use_amp:
                    with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                        pred = self.forward(x, x_mark, sample_ids=sample_ids, mode='train')

                        pred_raw, real_raw = self._restore_to_raw(
                            pred, label, dataModule.mean, dataModule.std
                        )

                        # 为了数值稳定，loss 用 float32 更稳（尤其是 cumsum 后的序列）
                        loss = compute_loss(self, x, pred_raw.float(), real_raw.float(), self.config)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    pred = self.forward(x, x_mark, dec_input=dec_input, sample_ids=sample_ids, mode='train')

                    pred_raw, real_raw = self._restore_to_raw(
                        pred, label, dataModule.mean, dataModule.std
                    )
                    loss = compute_loss(self, x, pred_raw, real_raw, self.config)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                sample_count += x.size(0)
        

        except Exception as e:
            print(f"[Train Error] Epoch {self.current_epoch} failed: {str(e)}")

        finally:
            self.eval()
            torch.set_grad_enabled(False)
            t2 = time.time()

            avg_loss = total_loss / sample_count if sample_count > 0 else 0
            print(f"[Train] Epoch {self.current_epoch} finished | "
                  f"Avg Loss: {avg_loss:.6f} | Time Cost: {t2-t1:.2f}s")
        return avg_loss, t2 - t1

    def valid(self, dataModule):

        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.val_data_loader 

        preds, reals, val_loss = [], [], 0.0

        ctx = torch.autocast(device_type=self.device_type, dtype=torch.float16) if self.config.use_amp else contextlib.nullcontext()

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids = batch
                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()
                # decoder input
                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat([label[:, :self.label_len, :], dec_input], dim=1).float().to(self.config.device)

                pred = self.forward(x, x_mark, dec_input=dec_input, sample_ids=sample_ids, mode='valid')

                pred_raw, real_raw = self._restore_to_raw(pred, label, dataModule.mean, dataModule.std)
                loss_item = compute_loss(self, x, pred_raw.float(), real_raw.float(), self.config)
                val_loss += loss_item.item() if torch.is_tensor(loss_item) else float(loss_item)

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        pred_raw, real_raw = self._restore_to_raw(preds, reals, dataModule.mean, dataModule.std)

        # 你原来的评估方式保持：真实值用 reals[:,:,-1]，预测用 pred_raw
        return ErrorMetrics(reals[:, :, -1], pred_raw, self.config, 'valid')

    def test(self, dataModule):

        self.eval()
        torch.set_grad_enabled(False)
        dataloader = dataModule.test_data_loader

        preds, reals = [], []

        ctx = torch.autocast(device_type=self.device_type, dtype=torch.float16) if self.config.use_amp else contextlib.nullcontext()

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids = batch
                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()
                # decoder input
                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat([label[:, :self.label_len, :], dec_input], dim=1).float().to(self.config.device)


                pred = self.forward(x, x_mark, dec_input=dec_input, sample_ids=sample_ids, mode='test')

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        pred_raw, real_raw = self._restore_to_raw(preds, reals, dataModule.mean, dataModule.std)

        return ErrorMetrics(reals[:, :, -1], pred_raw, self.config, 'test')

    #
    
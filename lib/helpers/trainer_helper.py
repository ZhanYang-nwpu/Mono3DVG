import os
import tqdm

import torch
import numpy as np

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint, load_detr
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 val_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_val_loss = 99999
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mono3dvg_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint_latest.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        elif cfg.get('detr_model', None):
            detr_model_path = os.path.join('./configs/' + cfg['detr_model'])
            assert os.path.exists(detr_model_path)
            load_detr(
                model=self.model.to(self.device),
                filename=detr_model_path,
                map_location=self.device,
                logger=self.logger
            )


    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_val_loss = self.best_val_loss
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                ckpt_name = os.path.join(self.output_dir, 'checkpoint_latest')
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),ckpt_name)
                # self.logger.info("Val Epoch {}".format(self.epoch))

                if self.tester is not None:
                    # val one epoch
                    cur_acc25, cur_acc5, cur_loss = self.tester.evaluate(epoch)
                    cur_result = cur_acc25 + cur_acc5

                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))
                else:
                    # val one epoch
                    cur_val_loss = self.val_one_epoch(epoch)
                    if cur_val_loss < best_val_loss:
                        best_val_loss = cur_val_loss
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch, best_val_loss, best_epoch),ckpt_name)
                    self.logger.info("Best Loss:{}, epoch:{}".format(best_val_loss, best_epoch))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        # print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)

            captions = targets["text"]
            im_name = targets['image_id']
            instanceID = targets['instance_id']
            ann_id = targets['anno_id']

            for key in targets.keys():
                if key not in ['image_id','text']:
                    targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            targets = self.prepare_targets(targets, inputs.shape[0])

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, calibs, img_sizes, captions, im_name, instanceID, ann_id)

            mono3dvg_losses_dict = self.mono3dvg_loss(outputs, targets)

            weight_dict = self.mono3dvg_loss.weight_dict
            mono3dvg_losses_dict_weighted = [mono3dvg_losses_dict[k] * weight_dict[k] for k in mono3dvg_losses_dict.keys() if k in weight_dict]
            mono3dvg_losses = sum(mono3dvg_losses_dict_weighted)

            mono3dvg_losses_dict = misc.reduce_dict(mono3dvg_losses_dict)
            mono3dvg_losses_dict_log = {}
            mono3dvg_losses_log = 0
            for k in mono3dvg_losses_dict.keys():
                if k in weight_dict:
                    mono3dvg_losses_dict_log[k] = (mono3dvg_losses_dict[k] * weight_dict[k]).item()
                    mono3dvg_losses_log += mono3dvg_losses_dict_log[k]
            mono3dvg_losses_dict_log["loss_mono3dvg"] = mono3dvg_losses_log

            flags = [True] * 5
            if batch_idx % 30 == 0:
                print_str = 'Epoch: [{}][{}/{}]\t' \
                            'Loss_mono3dvg: {:.2f}\t' \
                            'loss_ce: {:.2f}\t' \
                            'loss_bbox: {:.2f}\t' \
                            'loss_giou: {:.2f}\t' \
                            'loss_depth: {:.2f}\t' \
                            'loss_dim: {:.2f}\t' \
                            'loss_angle: {:.2f}\t' \
                            'loss_center: {:.2f}\t' \
                            'loss_depth_map: {:.2f}\t' \
                    .format(\
                    epoch, batch_idx, len(self.train_loader), \
                    mono3dvg_losses_dict_log["loss_mono3dvg"], \
                    mono3dvg_losses_dict_log['loss_ce'], \
                    mono3dvg_losses_dict_log['loss_bbox'], \
                    mono3dvg_losses_dict_log['loss_giou'], \
                    mono3dvg_losses_dict_log['loss_depth'], \
                    mono3dvg_losses_dict_log['loss_dim'],\
                    mono3dvg_losses_dict_log['loss_angle'], \
                    mono3dvg_losses_dict_log['loss_center'], \
                    mono3dvg_losses_dict_log['loss_depth_map'],\
                    )

                # print(print_str)
                self.logger.info(print_str)

            mono3dvg_losses.backward()
            self.optimizer.step()

            progress_bar.update()
        progress_bar.close()
        self.logger.info("Final Training Loss: " + str(mono3dvg_losses_dict_log["loss_mono3dvg"]))


    def val_one_epoch(self, epoch):
        torch.set_grad_enabled(False)
        self.model.eval()

        progress_bar = tqdm.tqdm(total=len(self.val_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.val_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)

            captions = targets["text"]
            im_name = targets['image_id']
            instanceID = targets['instance_id']
            ann_id = targets['anno_id']

            for key in targets.keys():
                if key not in ['image_id','text']:
                    targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            targets = self.prepare_targets(targets, inputs.shape[0])

            outputs = self.model(inputs, calibs, img_sizes, captions,im_name, instanceID, ann_id)

            mono3dvg_losses_dict = self.mono3dvg_loss(outputs, targets)

            weight_dict = self.mono3dvg_loss.weight_dict

            mono3dvg_losses_dict = misc.reduce_dict(mono3dvg_losses_dict)
            mono3dvg_losses_dict_log = {}
            mono3dvg_losses_log = 0
            for k in mono3dvg_losses_dict.keys():
                if k in weight_dict:
                    mono3dvg_losses_dict_log[k] = (mono3dvg_losses_dict[k] * weight_dict[k]).item()
                    mono3dvg_losses_log += mono3dvg_losses_dict_log[k]
            mono3dvg_losses_dict_log["loss_mono3dvg"] = mono3dvg_losses_log

            if batch_idx % 30 == 0:
                print_str = 'Evaluation: [{}][{}/{}]\t' \
                            'Loss_mono3dvg: {:.2f}\t' \
                            'loss_ce: {:.2f}\t' \
                            'loss_bbox: {:.2f}\t' \
                            'loss_giou: {:.2f}\t' \
                            'loss_depth: {:.2f}\t' \
                            'loss_dim: {:.2f}\t' \
                            'loss_angle: {:.2f}\t' \
                            'loss_center: {:.2f}\t' \
                            'loss_depth_map: {:.2f}\t' \
                    .format(\
                    epoch, batch_idx, len(self.val_loader), \
                    mono3dvg_losses_dict_log["loss_mono3dvg"], \
                    mono3dvg_losses_dict_log['loss_ce'], \
                    mono3dvg_losses_dict_log['loss_bbox'], \
                    mono3dvg_losses_dict_log['loss_giou'], \
                    mono3dvg_losses_dict_log['loss_depth'], \
                    mono3dvg_losses_dict_log['loss_dim'],\
                    mono3dvg_losses_dict_log['loss_angle'], \
                    mono3dvg_losses_dict_log['loss_center'], \
                    mono3dvg_losses_dict_log['loss_depth_map'],\
                    )

                # print(print_str)
                self.logger.info(print_str)

            progress_bar.update()
        progress_bar.close()
        return mono3dvg_losses_dict_log["loss_mono3dvg"]


    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list


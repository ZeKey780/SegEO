import numpy as np
import torch
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
import os

set_determinism(123)

data_dir = "./data/fullres/train1"
logdir = f"./logs/segEO_aiib"

model_save_path = os.path.join(logdir, "model")
augmentation = True

env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:1"
roi_size = [128, 128, 128]


def func(m, epochs):
    return np.exp(-10 * (1 - m / epochs) ** 2)


class AIIBTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.best_iou = None
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        self.augmentation = augmentation
        from model.segEO import SegEO

        self.model = SegEO(in_chans=1,  # 单个模态输入
                              out_chans=2,
                              depths=[2, 2, 2, 2],
                              feat_size=[48, 96, 192, 384])

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.train_process = 18
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)

        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()

    def load_model(self, checkpoint_path):

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            print(f"Checkpoint path {checkpoint_path} does not exist!")

    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)
        loss = self.cross(pred, label)
        self.log("training_loss", loss, step=self.global_step)
        return loss

    def convert_labels(self, labels):
        """将标签转换为二分类格式"""
        return (labels == 1).float()  # 只关注 `1=CT` 类别

    def get_input(self, batch):
        """获取图像和标签"""
        image = batch["data"]
        label = batch["seg"]
        return image, label[:, 0].long()  # 取 `seg` 第一维度作为标签

    def cal_iou(self, gt, pred):
        """计算 IoU"""
        intersection = np.sum((gt == 1) & (pred == 1))
        union = np.sum((gt == 1) | (pred == 1))
        return intersection / (union + 1e-6)

    def cal_dlr(self, gt, pred):
        """计算 DLR"""
        detected_length_gt = np.sum(gt == 1)
        detected_length_pred = np.sum(pred == 1)
        return detected_length_pred / (detected_length_gt + 1e-6)

    def cal_dbr(self, gt, pred, iou_threshold=0.8):
        # 计算每个分支的 IoU
        pred_branches = np.unique(pred)  # 假设每个分支有唯一的标签
        gt_branches = np.unique(gt)  # 假设每个分支有唯一的标签
        detected_branches = 0

        for branch in gt_branches:
            if branch == 0:  # 忽略背景
                continue
            gt_branch = (gt == branch)
            pred_branch = (pred == branch)
            iou = self.cal_iou(gt_branch, pred_branch)
            if iou >= iou_threshold:
                detected_branches += 1

        return detected_branches / len(gt_branches) if len(gt_branches) > 0 else 0
    #    def cal_dbr(self, gt, pred):
    #        """计算 DBR"""
    #        detected_branches_gt = np.sum((gt == 1) & (pred == 1))
    #        total_branches_gt = np.sum(gt == 1)
    #        return detected_branches_gt / (total_branches_gt + 1e-6)

    def validation_step(self, batch):
        """计算 IoU, DLR, DBR"""
        image, label = self.get_input(batch)
        output = self.model(image)
        output = output.argmax(dim=1)  # 取出类别预测值
        output = self.convert_labels(output)
        label = self.convert_labels(label)

        output = output.cpu().numpy()
        target = label.cpu().numpy()

        iou = self.cal_iou(target, output)
        dbr = self.cal_dbr(target, output)
        dlr = self.cal_dlr(target, output)

        return iou, dbr, dlr

    def validation_end(self, val_outputs):
        """计算均值，并保存最优模型"""
        ious, dbrs, dlrs = val_outputs

        mean_iou = ious.mean()
        mean_dbr = dbrs.mean()
        mean_dlr = dlrs.mean()
        mean_metric = (mean_iou + mean_dbr + mean_dlr) / 3  # 计算均值
        print(f"\nEpoch {self.epoch} Results:")
        print(f"IoU: {mean_iou:.4f}")
        print(f"DBR: {mean_dbr:.4f}")
        print(f"DLR: {mean_dlr:.4f}")

        self.log("mean_iou", mean_iou, step=self.epoch)
        self.log("mean_dbr", mean_dbr, step=self.epoch)
        self.log("mean_dlr", mean_dlr, step=self.epoch)

        if mean_iou > self.best_iou:
            self.best_iou = mean_iou
            model_path = os.path.join(model_save_path, f"epoch{self.epoch}_{mean_metric:.4f}_{mean_iou:.4f}.pt")
            print(f"Saving model: {model_path}")
            torch.save(self.model.state_dict(), model_path)

        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                                    f"final_model_{self.epoch}_{mean_iou:.4f}.pt"),
                                       delete_symbol="final_model")
        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{mean_iou:.4f}.pt"))

        print(f"mean_iou is {mean_iou}, mean_dbr is {mean_dbr}, mean_dlr is {mean_dlr}")



if __name__ == "__main__":
    trainer = AIIBTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17759,
                           training_script=__file__)

    #    checkpoint_path = os.path.join(model_save_path, "epoch175_0.9281_0.8947.pt")  # 替换为保存的权重文件路径
    ##    if checkpoint_path is not None:
    #    trainer.load_model(checkpoint_path)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

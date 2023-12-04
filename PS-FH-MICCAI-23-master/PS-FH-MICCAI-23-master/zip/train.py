from pathlib import Path
from torch.utils.data import Dataset
import SimpleITK
import matplotlib.pyplot as plt
from segment_anything.modeling.mask_decoder import MLP, MaskDecoder
import numpy as np
import sys
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.losses import DiceLoss
from importlib import import_module
from segment_anything import sam_model_registry
import torch
import logging
import os
import torch.nn as nn
import torch.nn.functional as F


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.6):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch.long())
    loss_dice = dice_loss(low_res_logits.argmax(dim=1), low_res_label_batch)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


class Fetal_dataset(Dataset):
    def __init__(self, list_dir, transform=None):
        self.transform = transform  # using transform in torch!
        images = [SimpleITK.GetArrayFromImage(pkg2.resize_image_itk(SimpleITK.ReadImage(str(i)), (512, 512, 3))) for i
                  in
                  list_dir[0]]
        labels = [SimpleITK.GetArrayFromImage(pkg2.resize_image_itk(SimpleITK.ReadImage(str(i)), (512, 512))) for i in
                  list_dir[1]]
        self.images = np.array(images)
        self.labels = np.array(labels)
        print("-" * 20)
        print(self.images.shape)
        print(self.labels.shape)
        print("-" * 20)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, mask = pkg2.correct_dims(self.images[idx].transpose((1, 2, 0)), self.labels[idx])
        sample = {}
        if self.transform:
            image, mask, low_mask = self.transform(image, mask)

        sample['image'] = image
        sample['low_res_label'] = low_mask.unsqueeze(0)
        sample['label'] = mask.unsqueeze(0)

        return sample


class MyDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        intersection = (y_pred[:, 1:2] * y_truth[:, 1:2]).sum() + (y_pred[:, 2:] * y_truth[:, 2:]).sum()
        union = y_pred[:, 1:2].sum() + y_pred[:, 2:].sum() + y_truth[:, 1:2].sum() + y_truth[:, 2:].sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_score.requires_grad_(True)

        dice1 = (2. * ((y_pred[:, 1:2] * y_truth[:, 1:2]).sum()) + self.smooth) / (
                y_pred[:, 1:2].sum() + y_truth[:, 1:2].sum() + self.smooth)
        dice2 = (2. * ((y_pred[:, 2:] * y_truth[:, 2:]).sum()) + self.smooth) / (
                y_pred[:, 2:].sum() + y_truth[:, 2:].sum() + self.smooth)
        dice1.requires_grad_(False)
        dice2.requires_grad_(False)
        return 1 - dice_score, dice1, dice2


class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self,dice_weight=0.6):
        super(Mask_DC_and_BCE_loss, self).__init__()
        self.dc = MyDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        target = F.one_hot(target.squeeze(1).long(), 3)

        low_res_logits_t = torch.flatten(low_res_logits.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
        target_t = torch.flatten(target, start_dim=0, end_dim=2)
        loss_ce = F.cross_entropy(low_res_logits_t, target_t.argmax(dim=1))
        loss_dice, _, __ = self.dc(F.one_hot(torch.argmax(low_res_logits_t, dim=1), 3), target_t)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss, loss_ce, loss_dice


if __name__ == "__main__":
    pkg2 = import_module('data_us')
    tf_train = pkg2.JointTransform2D(img_size=512, low_img_size=128,
                                     ori_size=256, crop=None, p_flip=0.5, p_rota=0.5, p_scale=0.0, p_gaussn=0.0,
                                     p_contr=0.0, p_gama=0.0, p_distor=0.0, color_jitter_params=None,
                                     long_mask=True)  # image reprocessing
    tf_val = pkg2.JointTransform2D(img_size=512, low_img_size=128, ori_size=256, crop=None, p_flip=0.0,
                                   color_jitter_params=None,
                                   long_mask=True)

    root_path = Path('./dataset')
    image_files = np.array([(root_path / Path("image_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 4001)])
    label_files = np.array([(root_path / Path("label_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 4001)])

    base_lr = 0.001
    num_classes = 2
    batch_size = 4
    multimask_output = True
    warmup = 1
    max_epoch = 30
    save_interval = 5
    warmup_period = 500
    weight_decay = 7
    device = 0
    devices = [0, ]
    fold_n = 4

    with open("./train_fold_4.txt", "r") as file:
        lines = file.readlines()
    train_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    with open("./val_fold_4.txt", "r") as file:
        lines = file.readlines()
    test_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    print(len(train_index), len(test_index))

    snapshot_path = f'./train_ckpts/b{batch_size}_wd{weight_decay}_results/{fold_n}'
    os.makedirs(snapshot_path, exist_ok=True)

    db_train = Fetal_dataset(transform=tf_train,
                             list_dir=(image_files[np.array(train_index)], label_files[np.array(train_index)]))
    db_val = Fetal_dataset(transform=tf_val,
                           list_dir=(image_files[np.array(test_index)], label_files[np.array(test_index)]))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    sam, img_embedding_size = sam_model_registry['vit_h'](image_size=512,
                                                          num_classes=8,
                                                          checkpoint='./checkpoints/sam_vit_h_4b8939.pth',
                                                          pixel_mean=[0, 0, 0],
                                                          pixel_std=[1, 1, 1])
    pkg = import_module('sam_lora_image_encoder')
    net = pkg.LoRA_Sam(sam, 4).to(device)
    ckpts = sorted(os.listdir("./ckpts"), key=lambda x: int(x.split("_")[-1].replace(".pth", "")))
    net.sam.mask_decoder.iou_prediction_head = MLP(
        net.sam.mask_decoder.transformer_dim, 256, num_classes + 1, 3
    )
    net.sam.mask_decoder = MaskDecoder(transformer=net.sam.mask_decoder.transformer,
                                       transformer_dim=net.sam.mask_decoder.transformer_dim,
                                       num_multimask_outputs=num_classes
                                       )
    net.load_lora_parameters(f'./ckpts/{ckpts[-1]}')
    net = torch.nn.DataParallel(net, device_ids=devices)

    model = net
    model.to(device)
    print("The length of train set is: {}".format(len(db_train)))

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    max_iterations = max_epoch * len(trainloader)

    if warmup:
        b_lr = base_lr / warmup_period
    else:
        b_lr = base_lr

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999),
                            weight_decay=0.1)
    iter_num = 0
    writer = SummaryWriter(f'results/{fold_n}/logs')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 100
    iterator = range(max_epoch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_f = Mask_DC_and_BCE_loss()
    for epoch_num in iterator:
        train_loss_ce = []
        train_loss_dice = []

        val_loss_ce = []
        val_loss_dice = []
        val_dice_score = []
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            # print(image_batch.shape, label_batch.shape, low_res_label_batch.shape)
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            low_res_label_batch = low_res_label_batch.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(image_batch, multimask_output, 512)
                loss, loss_ce, loss_dice = loss_f(outputs, low_res_label_batch)
            # print(f"loss:{loss.detach().cpu().numpy():.8f}, loss_ce:{loss_ce.detach().cpu().numpy():.8f}, loss_dice:{loss_dice.detach().cpu().numpy():.8f}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if warmup and iter_num < warmup_period:
                lr_ = base_lr * ((iter_num + 1) / warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if warmup:
                    shift_iter = iter_num - warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (
                        1.0 - shift_iter / max_iterations) ** weight_decay  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iter_num)
            iter_num = iter_num + 1
            train_loss_ce.append(loss_ce.detach().cpu().numpy())
            train_loss_dice.append(loss_dice.detach().cpu().numpy())

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f ,lr:%f' % (
                iter_num, loss.item(), loss_ce.item(), loss_dice.item(),
                optimizer.param_groups[0]['lr']))

        train_loss_ce_mean = np.mean(train_loss_ce)
        train_loss_dice_mean = np.mean(train_loss_dice)

        writer.add_scalar('info/total_loss', train_loss_ce_mean + train_loss_dice_mean, iter_num)
        writer.add_scalar('info/loss_ce', train_loss_ce_mean, iter_num)
        writer.add_scalar('info/loss_dice', train_loss_dice_mean, iter_num)
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                image_batch, label_batch = sampled_batch["image"].to(device), sampled_batch["label"].to(
                    device)
                low_res_label_batch = sampled_batch['low_res_label']
                low_res_label_batch = low_res_label_batch.to(device)

                outputs = model(image_batch, multimask_output, 512)
                loss, loss_ce, loss_dice = loss_f(outputs, low_res_label_batch)

                val_loss_ce.append(loss_ce.detach().cpu().numpy())
                val_loss_dice.append(loss_dice.detach().cpu().numpy())
                """
                if i_batch % 100 == 0:
                    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                    ax[0].imshow(sampled_batch['image'][0].cpu().numpy().transpose(1, 2, 0))
                    ax[0].set_title('image')
                    ax[1].imshow(sampled_batch['label'][0][0])
                    ax[1].set_title('label')
                    output_masks = outputs['masks']
                    output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)

                    ax[2].imshow(output_masks[0].cpu()[0])
                    ax[2].set_title('prediction')
                    plt.show()
                """
            val_loss_ce_mean = np.mean(val_loss_ce)
            val_loss_dice_mean = np.mean(val_loss_dice)

            writer.add_scalar('info/val_total_loss', val_loss_ce_mean + val_loss_dice_mean, iter_num)
            writer.add_scalar('info/val_loss_ce', val_loss_ce_mean, iter_num)
            writer.add_scalar('info/val_loss_dice', val_loss_dice_mean, iter_num)

            logging.info('epoch %d : val loss : %f, val loss_ce: %f, val loss_dice: %f' % (
                epoch_num, val_loss_ce_mean + val_loss_dice_mean,
                val_loss_ce_mean, val_loss_dice_mean))

        if val_loss_dice_mean < best_performance:
            best_performance = val_loss_dice_mean
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num)+'dice_'+str(best_performance) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
        model.train()

    writer.close()

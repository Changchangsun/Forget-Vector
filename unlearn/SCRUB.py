import sys
import time

import torch

import utils

from .impl import iterative_unlearn
import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
sys.path.append(".")
from imagenet import get_x_y_from_data_dict
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss


from tqdm import tqdm
class MaskedDataset(Dataset):
    def __init__(self, forget_set, retain_set, mask):
        super(MaskedDataset, self).__init__()
        self.forget_set = forget_set
        self.retain_set = retain_set
        self.mask = mask
        self.forget_len = len(forget_set)
         # 验证数据长度一致
        assert len(mask) == len(forget_set) + len(retain_set), "Mask length must match combined dataset length."

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        if self.mask[idx] == 0:
            # 样本来自 forget_set
            image, target = self.forget_set[idx]
            source = 0  # 标识来自 forget_set
        else:
            # 样本来自 retain_set
            adjusted_idx = idx - len(self.forget_set)
            image, target = self.retain_set[adjusted_idx]
            source = 1  # 标识来自 retain_set

        return image, target, source



def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


@iterative_unlearn
# train_acc = unlearn_iter_func(
#                     data_loaders, model_t, model, criterion, optimizer, epoch, args, **kwargs
#                 )
def SCRUB(data_loaders, model_t, model, criterion, optimizer, epoch, args):

    s_optim = 'sgd'
    s_gamma = 0.99
    s_alpha = 0.001
    s_beta = 0
    s_smoothing = 0.0
    s_msteps = 2
    s_clip = 0.2
    s_sstart = 10
    s_kd_T = 4
    s_distill = 'kd'

    s_sgda_batch_size = 128
    s_del_batch_size = 32
    s_sgda_epochs = 3 ############################
    s_sgda_learning_rate = 0.0005
    s_lr_decay_epochs = [3,5,9]
    s_lr_decay_rate = 0.1
    s_sgda_weight_decay = 5e-4
    s_sgda_momentum = 0.9

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(s_kd_T)
    # criterion_kd = DistillKL(s_kd_T)

    # criterion_list = nn.ModuleList([])
    # criterion_list.append(criterion_cls)    # classification loss
    # criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    # criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # # optimizer
    # if s_optim == "sgd":
    #     optimizer = torch.optim.SGD(trainable_list.parameters(),
    #                       lr=s_sgda_learning_rate, ##0.0005
    #                       momentum=s_sgda_momentum, ##0.9
    #                       weight_decay=s_sgda_weight_decay)#5e-4
    
    forget_loader = data_loaders["forget"]
    remain_loader = data_loaders["retain"]
    
    # forget_set = forget_loader.dataset
    # retain_set = remain_loader.dataset
    
    # mask = [0] * len(forget_set) + [1] * len(retain_set)
    # combined_dataset = MaskedDataset(forget_set, retain_set, mask)
    # combined_dataset_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    print("SCRUB")
    
    # criterion_cls = criterion_list[0]
    # criterion_div = criterion_list[1]
    # criterion_kd = criterion_list[2]
    
    # losses = utils.AverageMeter()
    # top1 = utils.AverageMeter()

    # switch to train mode
    model.train()
    
    ###所以说前1,2 epoch会有在forget set上，然后retain set
    ##在第3个epoch，只只在retain set上
    ##所以我们把训练的epoch数设为3,；0,1,；2
    
    if epoch<args.scrub_forget_epoch:
        #先是forget set
        for i, (image, target) in enumerate(tqdm(forget_loader)):  
            image = image.cuda()
            target = target.cuda()
            logit_s = model(image)
            with torch.no_grad():
                logit_t = model_t(image)
            # cls + kl div
            # loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)
            
            loss = -loss_div
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    ##后是retain set
    for i, (image, target) in enumerate(tqdm(remain_loader)):  
        image = image.cuda()
        target = target.cuda()
        logit_s = model(image)
        with torch.no_grad():
            logit_t = model_t(image)
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        
        loss = s_gamma * loss_cls + s_alpha * loss_div
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 0


    #     output = output_clean.float()
    #     loss = loss.float()
    #     # measure accuracy and record loss
    #     prec1 = utils.accuracy(output.data, target)[0]

    #     losses.update(loss.item(), image.size(0))
    #     top1.update(prec1.item(), image.size(0))

    #     if (i + 1) % args.print_freq == 0:
    #         end = time.time()
    #         print(
    #             "Epoch: [{0}][{1}/{2}]\t"
    #             "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
    #             "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
    #             "Time {3:.2f}".format(
    #                 epoch, i, len(combined_dataset_loader), end - start, loss=losses, top1=top1
    #             )
    #         )
    #         start = time.time()

    # print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return 0

import sys
import time

import torch

import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from torch.utils.data import Dataset, DataLoader, ConcatDataset

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
def NegGrad_plus(data_loaders, model, criterion, optimizer, epoch, args):
    forget_loader = data_loaders["forget"]
    remain_loader = data_loaders["retain"]
    
    forget_set = forget_loader.dataset
    retain_set = remain_loader.dataset
    mask = [0] * len(forget_set) + [1] * len(retain_set)
    combined_dataset = MaskedDataset(forget_set, retain_set, mask)
    combined_dataset_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    # combined_dataset_loader = DataLoader(
    #     combined_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=4,  # 增加 num_workers 提高数据加载速度
    #     # pin_memory=True,  # 启用 pin_memory 加速数据传输到 GPU
    #     shuffle=True
    # )

    print("NegGrad_plus")
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    beta = args.beta_neggrad_plus
    
    for i, (image, target, source) in enumerate(tqdm(combined_dataset_loader)):  
        
        image = image.cuda()
        target = target.cuda()
        source = source.cuda()

        # compute output
        output_clean = model(image)
        num_retain = source.sum()
        
        ##pos
        # pos_position = source.bool()
        pos_position = (source == 1)
        target_select_r = target[pos_position]
        output_output_r = output_clean[pos_position]
        
        ##neg
        # neg_position = (1-source).bool()
        neg_position = (source == 0)
        target_select_f = target[neg_position]
        output_output_f = output_clean[neg_position]        
                
        loss = beta*criterion(output_output_r, target_select_r)/num_retain - (1-beta)*criterion(output_output_f, target_select_f)/(args.batch_size - num_retain)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


# @iterative_unlearn
# def GA_l1(data_loaders, model, criterion, optimizer, epoch, args):
#     train_loader = data_loaders["forget"]

#     losses = utils.AverageMeter()
#     top1 = utils.AverageMeter()

#     # switch to train mode
#     model.train()

#     start = time.time()
#     for i, (image, target) in enumerate(train_loader):
#         if epoch < args.warmup:
#             utils.warmup_lr(
#                 epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
#             )

#         image = image.cuda()
#         target = target.cuda()

#         # compute output
#         output_clean = model(image)
#         loss = -criterion(output_clean, target) + args.alpha * l1_regularization(model)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         output = output_clean.float()
#         loss = loss.float()
#         # measure accuracy and record loss
#         prec1 = utils.accuracy(output.data, target)[0]

#         losses.update(loss.item(), image.size(0))
#         top1.update(prec1.item(), image.size(0))

#         if (i + 1) % args.print_freq == 0:
#             end = time.time()
#             print(
#                 "Epoch: [{0}][{1}/{2}]\t"
#                 "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
#                 "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
#                 "Time {3:.2f}".format(
#                     epoch, i, len(train_loader), end - start, loss=losses, top1=top1
#                 )
#             )
#             start = time.time()

#     print("train_accuracy {top1.avg:.3f}".format(top1=top1))

#     return top1.avg

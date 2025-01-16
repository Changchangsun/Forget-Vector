import time
from copy import deepcopy

import numpy as np
import torch
import utils
from torch.autograd import grad

from .impl import iterative_unlearn


def get_batch_grad(loss, model):
    params = list([param for param in model.parameters() if param.requires_grad]
)
    batch_grad = grad(loss, params)
    batch_grad = torch.cat([x.view(-1) for x in batch_grad])
    return batch_grad
    
    
def get_avg_grad(sum_grad, loader, model, criterion):
    sum_grad  = None
    total_samples = 0
    # for batch in enumerate(loader):
    for it, (image, target) in enumerate(loader):

        # batch = [data.cuda() for data in batch]
        # input = batch[:-1]
        # label = batch[-1]
        
        image = image.cuda()
        target = target.cuda()
        output_clean = model(image)


        model.zero_grad()
        loss = criterion(output_clean, target)

        batch_grad_ = get_batch_grad(loss, model)

        if sum_grad is None:
            sum_grad = batch_grad_ * image.shape[0]
        else:
            sum_grad += batch_grad_ * image.shape[0]
        total_samples += image.shape[0]

    return sum_grad
        
def woodfisher(model, loader, perturb_vector, criterion):
    # device = self.model.device

    model.eval()
    k_vec = torch.clone(perturb_vector)
    N = 1000
    o_vec = None
    
    # for idx, batch in enumerate(loader):
    for idx, (image, target) in enumerate(loader):
        if idx > N:
            break  
    
        image = image.cuda()
        target = target.cuda()
        output_clean = model(image)

        model.zero_grad()
        loss = criterion(output_clean, target)

        batch_grad = get_batch_grad(loss, model)

        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(batch_grad)
            else:
                tmp = torch.dot(o_vec, batch_grad)
                k_vec -= (torch.dot(k_vec, batch_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        # if idx > N:
        #     return k_vec
    return k_vec
            
@iterative_unlearn            
def IU(data_loaders, model, criterion, optimizer, epoch, args, mask=None):

    print("Start IU...")
    
    forget_loader = data_loaders["forget"]
    remain_loader = data_loaders["retain"]
    params = [param for param in model.parameters() if param.requires_grad]
    
    # forget_grad = torch.zeros_like(torch.cat(params)).cuda()
    # remain_grad = torch.zeros_like(torch.cat(params)).cuda()
    forget_grad = torch.zeros_like(torch.cat([param.view(-1) for param in params])).cuda()
    remain_grad = torch.zeros_like(torch.cat([param.view(-1) for param in params])).cuda()

    
    n_forget = len(forget_loader.dataset)
    n_remain = len(remain_loader.dataset)
    
    print("Compute gradient.")
    
    forget_grad = get_avg_grad(forget_grad, forget_loader, model, criterion)
    remain_grad = get_avg_grad(remain_grad, remain_loader, model, criterion)
    
    remain_grad *= n_forget / ((n_forget + n_remain) * n_remain)
    forget_grad /= n_forget + n_remain
    
    remain_loader_single = torch.utils.data.DataLoader(remain_loader.dataset, batch_size=1, shuffle=False)
    
    perturb = woodfisher(model, remain_loader_single, forget_grad - remain_grad, criterion)

    v = args.alpha_iu * perturb
    curr = 0
    with torch.no_grad():
        for param in [param for param in model.parameters() if param.requires_grad]:
            length = param.numel()
            param += v[curr : curr + length].view(param.shape)
            curr += length
    return model


import copy
import os
from collections import OrderedDict
import gc
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import arg_parser
import evaluation
import unlearn
import delta_utils
from delta_imagenet import get_x_y_from_data_dict
from trainer import validate,delta_validate
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from evaluation.mia import *
from datetime import datetime

# Define optimizer for the perturbation
import time

from torch.utils.data import Dataset, DataLoader, ConcatDataset
import wandb

import distributed as dist

from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

            
class MaskedDataset(Dataset):
    def __init__(self, forget_set, retain_set, mask):
        super(MaskedDataset, self).__init__()
        self.forget_set = forget_set
        self.retain_set = retain_set
        self.mask = mask
        self.forget_len = len(forget_set)
        assert len(mask) == len(forget_set) + len(retain_set), "Mask length must match combined dataset length."

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        if self.mask[idx] == 0:
            image, target = self.forget_set[idx]
            source = 0  
        else:
            adjusted_idx = idx - len(self.forget_set)
            image, target = self.retain_set[adjusted_idx]
            source = 1 

        return image, target, source


def main():
    args = arg_parser.parse_args()

    # dist.init()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    exp_name_new = None
    if args.multi_classes_to_replace is not None:
        forget_classes = "_".join(str(i) for i in args.multi_classes_to_replace)
        exp_name_new  = "tau_"+str(args.Tau)+"_2losses_from_"+str(args.init_delta)+"_lr_"+str(args.unlearn_lr)+"_ExponentialLR_alpha_"+str(args.Alpha)+"_beta_"+str(args.Beta)+"_lambda_"+str(args.Lambda)+"_forgetclass_"+forget_classes+"_"+args.all_delta+"_"+str(args.train_seed)
    else:
        exp_name_new  = "tau_"+str(args.Tau)+"_2losses_from_"+str(args.init_delta)+"_lr_"+str(args.unlearn_lr)+"_ExponentialLR_alpha_"+str(args.Alpha)+"_beta_"+str(args.Beta)+"_lambda_"+str(args.Lambda)+"_"+args.all_delta+"_schedule"+str(args.schedule)+"_"+str(args.train_seed)


    result_path_new = os.path.join(args.result_path, exp_name_new)
    os.makedirs(result_path_new, exist_ok=True)
    
    
    delta_utils.setup_seed(args.train_seed)
    seed = args.train_seed
    # prepare dataset
    print("Start prepare dataset.")
    model = None
    unlearn_data_loaders = None
        
    
    if args.multi_classes_to_replace is not None:
        print("###############multi_classes_to_replace",args.multi_classes_to_replace)
        print("###class-wise")
        model, retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, val_retain_set, val_forget_set = delta_utils.setup_model_dataset(args)
        unlearn_data_sets = OrderedDict(
        retain = retain_set, forget = forget_set, retain_for_test = retain_for_test_set, forget_for_test = forget_for_test_set,
        val = val_set, val_retain = val_retain_set, val_forget = val_forget_set,
    )
    elif args.class_to_replace is not None and args.num_indexes_to_replace is None:
        print("###class-wise")
        model, retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, val_retain_set, val_forget_set, retain_adv_set, forget_adv_set, val_adv_set, val_retain_adv_set, val_forget_adv_set = delta_utils.setup_model_dataset(args)
        unlearn_data_sets = OrderedDict(
        retain = retain_set, forget = forget_set, retain_for_test = retain_for_test_set, forget_for_test = forget_for_test_set,
        val = val_set, val_retain = val_retain_set, val_forget = val_forget_set,
        retain_adv = retain_adv_set, forget_adv = forget_adv_set, val_adv = val_adv_set, val_retain_adv = val_retain_adv_set, val_forget_adv = val_forget_adv_set
    )
    elif args.class_to_replace is None and args.num_indexes_to_replace is not None:
        print("$$$$data-wise forget model") ##################
        model, retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, retain_adv_set, forget_adv_set, val_adv_set = delta_utils.setup_model_dataset(args)
        unlearn_data_sets = OrderedDict(
        retain = retain_set, forget = forget_set, retain_for_test = retain_for_test_set, forget_for_test = forget_for_test_set,
        val = val_set, retain_adv = retain_adv_set, forget_adv = forget_adv_set, val_adv = val_adv_set
    )
    elif args.class_to_replace is None and args.num_indexes_to_replace is None:#original
        print("$$$$$original model")
        model, train_set, train_for_test_set, val_set = delta_utils.setup_model_dataset(args)
        unlearn_data_sets = OrderedDict(
        retain=train_set, retain_for_test=train_for_test_set, val=val_set
    )

    model = model.to(device)
    
    res_clean = {}
    res = {}    
    
    if args.phase == "train":

        criterion = nn.CrossEntropyLoss(reduction='sum')
        print("Start train.")
        
        model = unlearn.load_original_checkpoint(model, device, 199, args)

        # Freeze the parameters of layer1
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        # version 1
        if args.all_delta == "all" and args.dataset == "imagenet10_transfer2":
            path_0 = "path/class0/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_10.0_lambda_1.0/delta_epoch_160.pth"
            path_1 = "path/class1/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            path_2 = "path/class2/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            path_3 = "path/class3/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            path_4 = "path/class4/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            path_5 = "path/class5/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            path_6 = "path/class6/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            path_7 = "path/class7/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            path_8 = "path/class8/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            path_9 = "path/class9/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1/delta_epoch_190.pth"
            
            delta_0 = torch.load(path_0).to(device)
            delta_1 = torch.load(path_1).to(device)
            delta_2 = torch.load(path_2).to(device)
            delta_3 = torch.load(path_3).to(device)
            delta_4 = torch.load(path_4).to(device)
            delta_5 = torch.load(path_5).to(device)
            delta_6 = torch.load(path_6).to(device)
            delta_7 = torch.load(path_7).to(device)
            delta_8 = torch.load(path_8).to(device)
            delta_9 = torch.load(path_9).to(device)
            deltas = [delta_0, delta_1, delta_2, delta_3, delta_4, delta_5, delta_6, delta_7, delta_8, delta_9]

            # weights = torch.randn(10, requires_grad=True)
            weights = torch.zeros(10, requires_grad=True)

            print(weights.is_leaf)  # This should print True
        
        # version 1
        elif args.all_delta == "all" and args.dataset == "cifar10_scc_transfer2":
            path_0 = "path/class0/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_3.0_lambda_1.0-best/delta_epoch_160.pth"
            path_1 = "path/class1/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_5.0_lambda_1.0-best/delta_epoch_90.pth"
            path_2 = "path/class2/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_4.0_lambda_1.0-best/delta_epoch_110.pth"
            path_3 = "path/class3/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_5.0_lambda_1.0-best/delta_epoch_120.pth"
            path_4 = "path/class4/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_4.0_lambda_1.0-best/delta_epoch_120.pth"
            path_5 = "path/class5/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_6.0_lambda_1.0-best/delta_epoch_120.pth"
            path_6 = "path/class6/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_3.0_lambda_1.0-best/delta_epoch_120.pth"
            path_7 = "path/class7/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_7.0_lambda_1.0-best/delta_epoch_120.pth"
            path_8 = "path/class8/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_4.0_lambda_1.0-best/delta_epoch_190.pth"
            path_9 = "path/class9/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_10.0_lambda_1.0-best/delta_epoch_98.pth"

            
            delta_0 = torch.load(path_0).to(device)
            delta_1 = torch.load(path_1).to(device)
            delta_2 = torch.load(path_2).to(device)
            delta_3 = torch.load(path_3).to(device)
            delta_4 = torch.load(path_4).to(device)
            delta_5 = torch.load(path_5).to(device)
            delta_6 = torch.load(path_6).to(device)
            delta_7 = torch.load(path_7).to(device)
            delta_8 = torch.load(path_8).to(device)
            delta_9 = torch.load(path_9).to(device)
            deltas = [delta_0, delta_1, delta_2, delta_3, delta_4, delta_5, delta_6, delta_7, delta_8, delta_9]

            # weights = torch.randn(10, requires_grad=True)
            weights = torch.zeros(10, requires_grad=True)

            print(weights.is_leaf)  # This should print True
        
        
        elif args.all_delta == "all" and args.dataset == "cifar10_scc":
            print("args.all_delta == all and args.dataset == cifar10_scc:")
            path_0 = "path/class0/paras/new/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_3.0_lambda_1.0-line24-epoch80/delta_epoch_80.pth"
            path_1 = "path/class1/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_8.0_lambda_1.0-epoch60/delta_epoch_60.pth"
            path_2 = "path/class2/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_4.0_lambda_1.0-epoch80/delta_epoch_80.pth"
            path_3 = "path/class3/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_5.0_lambda_1.0-epoch120/delta_epoch_120.pth"
            path_4 = "path/class4/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_5.0_lambda_1.0-epoch70/delta_epoch_70.pth"
            path_5 = "path/class5/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_6.0_lambda_1.0-epoch12/delta_epoch_120.pth"
            path_6 = "path/class6/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_3.0_lambda_1.0-epoch60/delta_epoch_60.pth"
            path_7 = "path/class7/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_7.0_lambda_1.0-epoch120/delta_epoch_120.pth"    
            path_8 = "path/class8/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_4.0_lambda_1.0-epoch190/delta_epoch_190.pth"
            path_9 = "path/class9/paras/new/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_10.0_lambda_1.0-best/delta_epoch_61.pth"
            delta_0 = torch.load(path_0).to(device)
            delta_1 = torch.load(path_1).to(device)
            delta_2 = torch.load(path_2).to(device)
            delta_3 = torch.load(path_3).to(device)
            delta_4 = torch.load(path_4).to(device)
            delta_5 = torch.load(path_5).to(device)
            delta_6 = torch.load(path_6).to(device)
            delta_7 = torch.load(path_7).to(device)
            delta_8 = torch.load(path_8).to(device)
            delta_9 = torch.load(path_9).to(device)
            
            deltas = [delta_0, delta_1, delta_2, delta_3, delta_4, delta_5, delta_6, delta_7, delta_8, delta_9]
            
            for d in deltas:
                print(d.shape,"####")

            weights = None
            if args.init_delta=="0":
                weights = torch.zeros(10).to(device)
            elif args.init_delta=="random":
                weights = torch.randn(10).to(device)
            elif args.init_delta=="1":
                weights = torch.ones(10).to(device)
            weights.requires_grad_(True)
            
            print(weights.is_leaf)  # This should print True
            print("weights",weights, weights.shape)
            print("delta_0",delta_0.shape)#torch.Size([3, 32, 32])
            print("delta_1",delta_1.shape)
            
        elif args.all_delta == "all" and args.dataset == "imagenet10":
            print("args.all_delta == all and args.dataset == imagenet10:")
            path_0 = "path/class0/paras/new/tau_1_2losses_from_0_lr02_ExponentialLR_alpha_1_beta_10_Lambda_1-line4-epoch30/delta_epoch_30.pth"
            path_1 = "path/class1/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_5.0_lambda_1.0-epoch20/delta_epoch_20.pth"
            path_2 = "path/class2/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_10.0_lambda_1.0-epoch30/delta_epoch_30.pth"
            path_3 = "path/class3/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_10.0_lambda_1.0-epoch50/delta_epoch_50.pth"
            path_4 = "path/class4/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_5.0_lambda_1.0-epoch10/delta_epoch_10.pth"
            path_5 = "path/class5/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_10.0_lambda_1.0-epoch20/delta_epoch_20.pth"
            path_6 = "path/class6/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_5.0_lambda_1.0-epoch60/delta_epoch_60.pth"
            path_7 = "path/class7/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_10.0_lambda_1.0-epoch60/delta_epoch_60.pth"
            path_8 = "path/class8/paras/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_10.0_lambda_1.0-epoch30/delta_epoch_30.pth"
            path_9 = "path/class9/paras/new/tau_1.0_2losses_from_0_lr_0.2_ExponentialLR_alpha_1.0_beta_5.0_lambda_1.0-epoch30/delta_epoch_30.pth"
            delta_0 = torch.load(path_0).to(device)
            delta_1 = torch.load(path_1).to(device)
            delta_2 = torch.load(path_2).to(device)
            delta_3 = torch.load(path_3).to(device)
            delta_4 = torch.load(path_4).to(device)
            delta_5 = torch.load(path_5).to(device)
            delta_6 = torch.load(path_6).to(device)
            delta_7 = torch.load(path_7).to(device)
            delta_8 = torch.load(path_8).to(device)
            delta_9 = torch.load(path_9).to(device)
            
            deltas = [delta_0, delta_1, delta_2, delta_3, delta_4, delta_5, delta_6, delta_7, delta_8, delta_9]
            
            for d in deltas:
                print(d.shape,"####")


            weights = None
            if args.init_delta=="0":
                weights = torch.zeros(10).to(device)
            elif args.init_delta=="random":
                weights = torch.randn(10).to(device)
            elif args.init_delta=="1":
                weights = torch.ones(10).to(device)
            weights.requires_grad_(True)
            
        optimizer = torch.optim.SGD(
            [weights],
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.schedule == 1:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        else:
            pass
        
        Tau = args.Tau

        # Lambda = 1.0
        Alpha = args.Alpha
        Beta = args.Beta
        Lambda = args.Lambda

        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d-%H-%M")
        name = exp_name_new
        # wandb.init(project=name+current_time)
        
        ###train data
        mask = [0] * len(forget_set) + [1] * len(retain_set)
        combined_dataset = MaskedDataset(forget_set, retain_set, mask)
        combined_dataset_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
        # combined_dataset_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

        #evaluation data
        #####################################################################################Start EValuation
        retain_loader = DataLoader(
                    retain_for_test_set, batch_size=args.batch_size, num_workers=4, shuffle=False
                )
        forget_loader = DataLoader(
                        forget_for_test_set, batch_size=args.batch_size, num_workers=4, shuffle=False
                    )
            
        val_loader = DataLoader(
                        val_set, batch_size=args.batch_size, num_workers=4, shuffle=False
                    )
            
        unlearn_data_loaders_acc = OrderedDict(
            retain=retain_loader, 
            forget=forget_loader, 
            val = val_loader
                    )
        
        for epoch in range(0, args.unlearn_epochs):
            current_time = datetime.now()
            current_time = current_time.strftime("%Y-%m-%d %H:%M")
            print("epoch",epoch,"########################################",current_time,"Tau",Tau,"Alpha",Alpha,"Beta",Beta,"Lambda",Lambda,"exp_name",exp_name_new,)
            
            all_loss = 0.0
            all_loss_f = 0.0
            all_loss_r = 0.0
            all_loss_norm = 0.0
            num_elements_equal_to_neg_tau = 0
            num_elements_retain = 0
            
            for i, (image, target, source) in enumerate(tqdm(combined_dataset_loader, desc='main_delta_2loss_transfer2.py')):  

                delta = sum(w * d for w, d in zip(weights, deltas))

                image = image.to(device)
                image = image + delta
                target = target.to(device)
                source = source.to(device)
                output = model(image).to(device)
                
                target_logit = output[range(len(output)), target] 
                
                mask = torch.arange(output.size(1), device=device).unsqueeze(0) != target.unsqueeze(1)
                masked_output = output.masked_select(mask).view(len(output), output.size(1) - 1)
                other_logit = torch.max(masked_output, dim=1).values

                l_f = torch.max(target_logit - other_logit, torch.tensor(-Tau, device=device))* (1-source)
                num_elements_equal_to_neg_tau += (l_f == torch.tensor(-Tau, device=device)).sum().item()
            
                l_f = torch.sum(l_f)/image.shape[0]
                
                l_r_aw = torch.max(other_logit - target_logit, torch.tensor(-Tau, device=device))* source
                num_elements_retain += (l_r_aw == torch.tensor(-Tau, device=device)).sum().item()
            
                target_select = target[source.bool()]
                output_output = output[source.bool()]
                l_r = criterion(output_output, target_select)/image.shape[0]
                l_dist = torch.norm(weights, p=2)

                    
                loss = Alpha * l_r + Beta * l_f + Lambda * l_dist 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                all_loss += loss.item()  
                all_loss_f += l_f.item()
                all_loss_r += l_r.item()
                all_loss_norm += l_dist.item()
             
            if args.schedule == 1:
                scheduler.step()
            else:
                pass
            
            delta_l2_norm = torch.norm(delta, p=2)


            loss_file = open(os.path.join(result_path_new,"loss.txt"),"a")
            print("###################num_elements_equal_to_neg_tau/len(forget)",num_elements_equal_to_neg_tau,"/",len(forget_set))#
            print("###################num_elements_retain/len(retain)",num_elements_retain,"/",len(retain_set))#
            print('###################Epoch:[{0}]\t''Loss {loss:.4f}\t''all_loss_f {all_loss_f:.4f}\t''all_loss_r {all_loss_r:.4f}\t''delta_l2_norm {delta_l2_norm:.4f}\t'.format(epoch, loss=all_loss,all_loss_f = all_loss_f,all_loss_r = all_loss_r,delta_l2_norm=delta_l2_norm))
            loss_file.write(str("epoch")+str(epoch)+"\t"+str("all_loss")+"\t"+str(all_loss)+"\t"+str("all_loss_f")+"\t"+str(all_loss_f)+
                            "\t"+str("all_loss_r")+"\t"+str(all_loss_r)+"\t"+str("all_loss_norm")+"\t"+str(all_loss_norm)+"\t"+str("delta_l2_norm")+"\t"+str(delta_l2_norm)+"\t"+"\n")
            loss_file.close()
            
            class ModifiedDataset(torch.utils.data.Dataset):
                def __init__(self, original_dataset, delta):
                    self.original_dataset = original_dataset
                    self.delta = delta

                def __len__(self):
                    return len(self.original_dataset)

                def __getitem__(self, idx):
                    image, target = self.original_dataset[idx]
                    image = image.to(device)
                    self.delta = self.delta.to(device)
                    image = image + self.delta
                    return image, target
                
            
            delta = sum(w * d for w, d in zip(weights, deltas))
            modified_forget_set = ModifiedDataset(forget_for_test_set, delta)
            delta_forget_set_loader = torch.utils.data.DataLoader(modified_forget_set, batch_size=args.batch_size, shuffle=False)

            modified_retain_set = ModifiedDataset(retain_for_test_set, delta)
            delta_retain_set_loader = torch.utils.data.DataLoader(modified_retain_set, batch_size=args.batch_size, shuffle=False)

            modified_val_set = ModifiedDataset(val_set, delta)
            delta_val_set_loader = torch.utils.data.DataLoader(modified_val_set, batch_size=args.batch_size, num_workers=4, shuffle=False)
                
            unlearn_data_loaders_mia = OrderedDict(
            retain=retain_loader, 
            forget=forget_loader, 
            val = val_loader,
            retain_delta = delta_retain_set_loader, 
            forget_delta = delta_forget_set_loader, 
            val_delta = delta_val_set_loader
            )
            
    
            if epoch == 0:
                for name, loader in unlearn_data_loaders_acc.items():
                    val_acc = validate(loader, model, criterion, args)
                    val_acc = "{:.3f}".format(val_acc)
                    res_clean[name] = val_acc
                criterions = ["confidence"]
                for cri in criterions:
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    mia_efficacy = MIAEfficacy(cri)
                    iteration = 1 #
                    pattern = "datawise"
                    result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, device, pattern, seed)    
                    result = "{:.3f}".format(result)
                    res_clean[cri] = result
                print(res_clean)
            
            if epoch%1==0 and args.dataset=="cifar10_scc":
                
                acc_file = open(os.path.join(result_path_new,"acc.txt"),"a")
                acc_file.write(str("epoch")+str(epoch)+"\t")
                for name, loader in unlearn_data_loaders_acc.items():
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                # utils.dataset_convert_to_test(loader.dataset,args)
                    val_acc = delta_validate(loader, model, criterion, delta, args)
                    val_acc = "{:.3f}".format(val_acc)
                    print("epoch",epoch,"start testing",name,current_time,val_acc)
                    res[name] = val_acc
                    acc_file.write(name+"\t"+str(val_acc)+"\t")
                    
                criterions = ["confidence"]
                for cri in criterions:
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    mia_efficacy = MIAEfficacy(cri)
                    # Call the evaluate method
                    iteration = 1 #
                    pattern = "datawise_with_delta"
                    result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, device, pattern, seed)    
                    result = "{:.3f}".format(result)
                    res[cri] = result
                    print("mia",result)
                    acc_file.write(cri+"\t"+str(result)+"\t")
                acc_file.write("\n")
                acc_file.close()
                
                print("no delta",res_clean)  ####
                print("yes delta",res)  ####

                    
            elif epoch%1==0 and args.dataset=="imagenet10":
                
                acc_file = open(os.path.join(result_path_new,"acc.txt"),"a")
                acc_file.write(str("epoch")+str(epoch)+"\t")
                for name, loader in unlearn_data_loaders_acc.items():
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    print("epoch",epoch,"start testing",name,current_time)
                # utils.dataset_convert_to_test(loader.dataset,args)
                    val_acc = delta_validate(loader, model, criterion, delta, args)
                    val_acc = "{:.3f}".format(val_acc)
                    res[name] = val_acc
                    acc_file.write(name+"\t"+str(val_acc)+"\t")
                    
                criterions = ["confidence"]
                for cri in criterions:
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    mia_efficacy = MIAEfficacy(cri)
                    # Call the evaluate method
                    iteration = 1 #
                    pattern = "datawise_with_delta"
                    result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, device, pattern, seed)    
                    result = "{:.3f}".format(result)
                    res[cri] = result
                    acc_file.write(cri+"\t"+str(result)+"\t")
                acc_file.write("\n")
                acc_file.close()
                
                print("no delta",res_clean)  ####
                print("yes delta",res)  ####
                
            weights_cpu = weights.cpu()
            output_path = os.path.join(result_path_new,f'weights_epoch_{epoch}.pth')
            torch.save(weights_cpu, output_path)
            print(f"Delta saved for epoch {epoch}")


if __name__ == "__main__":
    main()
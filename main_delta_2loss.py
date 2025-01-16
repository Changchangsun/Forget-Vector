#<Forget Vector, 12/24/2024>
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
import time
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import wandb
import distributed as dist
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

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.result_path, exist_ok=True)
    
    exp_name_new = None
    if args.num_indexes_to_replace is not None:
        exp_name_new  = "tau_"+str(args.Tau)+"_2losses_from_"+str(args.init_delta)+"_lr_"+str(args.unlearn_lr)+"_ExponentialLR_alpha_"+str(args.Alpha)+"_beta_"+str(args.Beta)+"_lambda_"+str(args.Lambda)+"_percent_"+str(args.percent)+"_seed_"+str(args.train_seed)
    elif args.class_to_replace is not None:
        exp_name_new  = "tau_"+str(args.Tau)+"_2losses_from_"+str(args.init_delta)+"_lr_"+str(args.unlearn_lr)+"_ExponentialLR_alpha_"+str(args.Alpha)+"_beta_"+str(args.Beta)+"_lambda_"+str(args.Lambda)+"_class_"+str(args.class_to_replace)+"_seed_"+str(args.train_seed)

    delta_utils.setup_seed(args.train_seed)
    seed = args.train_seed
    model = None
    unlearn_data_loaders = None
        
    if args.class_to_replace is not None and args.num_indexes_to_replace is None:
        model, retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, val_retain_set, val_forget_set, retain_adv_set, forget_adv_set, val_adv_set, val_retain_adv_set, val_forget_adv_set = delta_utils.setup_model_dataset(args)
        unlearn_data_sets = OrderedDict(
        retain = retain_set, forget = forget_set, retain_for_test = retain_for_test_set, forget_for_test = forget_for_test_set,
        val = val_set, val_retain = val_retain_set, val_forget = val_forget_set,
        retain_adv = retain_adv_set, forget_adv = forget_adv_set, val_adv = val_adv_set, val_retain_adv = val_retain_adv_set, val_forget_adv = val_forget_adv_set
    )
    elif args.class_to_replace is None and args.num_indexes_to_replace is not None:
        model, retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, retain_adv_set, forget_adv_set, val_adv_set = delta_utils.setup_model_dataset(args)
        unlearn_data_sets = OrderedDict(
        retain = retain_set, forget = forget_set, retain_for_test = retain_for_test_set, forget_for_test = forget_for_test_set,
        val = val_set, retain_adv = retain_adv_set, forget_adv = forget_adv_set, val_adv = val_adv_set
    )
    elif args.class_to_replace is None and args.num_indexes_to_replace is None:
        model, train_set, train_for_test_set, val_set = delta_utils.setup_model_dataset(args)
        unlearn_data_sets = OrderedDict(
        retain=train_set, retain_for_test=train_for_test_set, val=val_set
    )
        
    model = model.to(device)
    
    res = {}    
    if args.phase == "train" and args.class_to_replace is not None:
        result_path_new = os.path.join(args.result_path, exp_name_new)
        os.makedirs(result_path_new, exist_ok=True)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        model = unlearn.load_original_checkpoint(model, device, 159, args)

        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        
        if args.init_delta == "random":
        #version 1: randomly initialize
            delta = torch.randn(unlearn_data_sets["retain"][0][0].shape).to(device)  
            delta_version = "from random"
        elif args.init_delta == "0":
        #version 2: start from zero, start from original input
            delta = torch.zeros_like(unlearn_data_sets["retain"][0][0]).to(device)  
            delta_version = "from 0"
        delta.requires_grad_(True)
        print("delta.shape",delta.shape)

        optimizer = torch.optim.SGD(
            [delta],
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        scheduler = None
        if args.schedule==1:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        else:
            print("no scheduler")

        Tau = args.Tau

        Alpha = args.Alpha
        Beta = args.Beta
        Lambda = args.Lambda

        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d-%H-%M")
        name = exp_name_new
        
        mask = [0] * len(forget_set) + [1] * len(retain_set)
        combined_dataset = MaskedDataset(forget_set, retain_set, mask)
        combined_dataset_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
        
        retain_loader = DataLoader(
                    retain_for_test_set, batch_size=args.batch_size, num_workers=2, shuffle=False
                )
        forget_loader = DataLoader(
                        forget_for_test_set, batch_size=args.batch_size, num_workers=2, shuffle=False
                    )
            
        val_loader = DataLoader(
                        val_set, batch_size=args.batch_size, num_workers=2, shuffle=False
                    )
            
        val_retain_loader = DataLoader(
                        val_retain_set, batch_size=args.batch_size, num_workers=2, shuffle=False
                    )
            
        val_forget_loader = DataLoader(
                        val_forget_set, batch_size=args.batch_size, num_workers=2, shuffle=False
                    )
        
        unlearn_data_loaders_acc = OrderedDict(
        retain=retain_loader, 
        forget=forget_loader, 
        val_retain=val_retain_loader, 
        val_forget=val_forget_loader
        )
        
        for epoch in range(0, args.unlearn_epochs+1):
            current_time = datetime.now()
            current_time = current_time.strftime("%Y-%m-%d %H:%M")            
            all_loss = 0.0
            all_loss_f = 0.0
            all_loss_r = 0.0
            all_loss_norm = 0.0
            num_elements_equal_to_neg_tau = 0
            num_elements_retain = 0

            start_time = time.time()
            for i, (image, target, source) in enumerate(tqdm(combined_dataset_loader, desc='main_delta_2loss.py')):  
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
                l_dist = torch.norm(delta, p=2)/image.shape[0]
                    
                loss = Alpha * l_r + Beta * l_f + Lambda * l_dist 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                all_loss += loss.item()  
                all_loss_f += l_f.item()
                all_loss_r += l_r.item()
                all_loss_norm += l_dist.item()          

            if args.schedule==1:
                scheduler.step()
            else:
                print("no scheduler")

            delta_l2_norm = torch.norm(delta, p=2)

            loss_file = open(os.path.join(result_path_new,"loss.txt"),"a")
            # print("###################num_elements_equal_to_neg_tau/len(forget)",num_elements_equal_to_neg_tau,"/",len(forget_set))
            # print("###################num_elements_retain/len(retain)",num_elements_retain,"/",len(retain_set))
            # print('###################Epoch:[{0}]\t''Loss {loss:.4f}\t''all_loss_f {all_loss_f:.4f}\t''all_loss_r {all_loss_r:.4f}\t''delta_l2_norm {delta_l2_norm:.4f}\t'.format(epoch, loss=all_loss,all_loss_f = all_loss_f,all_loss_r = all_loss_r,delta_l2_norm=delta_l2_norm))
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
                
            modified_forget_set = ModifiedDataset(forget_for_test_set, delta)
            delta_forget_set_loader = torch.utils.data.DataLoader(modified_forget_set, batch_size=args.batch_size, shuffle=False)

            modified_retain_set = ModifiedDataset(retain_for_test_set, delta)
            delta_retain_set_loader = torch.utils.data.DataLoader(modified_retain_set, batch_size=args.batch_size, shuffle=False)

            modified_val_retain_set = ModifiedDataset(val_retain_set, delta)
            delta_val_retain_set_loader = torch.utils.data.DataLoader(modified_val_retain_set, batch_size=args.batch_size,  shuffle=False)
                
            modified_val_forget_set = ModifiedDataset(val_forget_set, delta)
            delta_val_forget_set_loader = torch.utils.data.DataLoader(modified_val_forget_set, batch_size=args.batch_size, shuffle=False)
                
            unlearn_data_loaders_mia = OrderedDict(
            retain=retain_loader, 
            forget=forget_loader, 
            val_retain=val_retain_loader, 
            val_forget=val_forget_loader,
            retain_delta = delta_retain_set_loader, 
            forget_delta = delta_forget_set_loader, 
            val_retain_delta = delta_val_retain_set_loader, 
            val_forget_delta = delta_val_forget_set_loader
            )
            
    
            if epoch%10==0 and args.dataset=="cifar10":
                
                acc_file = open(os.path.join(result_path_new,"acc.txt"),"a")
                acc_file.write(str("epoch")+str(epoch)+"\t")
                for name, loader in unlearn_data_loaders_acc.items():
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    val_acc = delta_validate(loader, model, criterion, delta, args)
                    val_acc = "{:.3f}".format(val_acc)
                    res[name] = val_acc
                    acc_file.write(name+"\t"+str(val_acc)+"\t")
                    
                criterions = ["confidence"]
                for cri in criterions:
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    mia_efficacy = MIAEfficacy(cri)
                    iteration = 1 
                    pattern = "classwise_with_delta"
                    result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, device, pattern, seed)    
                    result = "{:.3f}".format(result)
                    res[cri] = result
                    acc_file.write(cri+"\t"+str(result)+"\t")
                acc_file.write("\n")
                acc_file.close()
                    
            elif epoch%10==0 and args.dataset=="imagenet10":
                
                acc_file = open(os.path.join(result_path_new,"acc.txt"),"a")
                acc_file.write(str("epoch")+str(epoch)+"\t")
                for name, loader in unlearn_data_loaders_acc.items():
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    val_acc = delta_validate(loader, model, criterion, delta, args)
                    val_acc = "{:.3f}".format(val_acc)
                    res[name] = val_acc
                    acc_file.write(name+"\t"+str(val_acc)+"\t")
                    
                criterions = ["confidence"]
                for cri in criterions:
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    mia_efficacy = MIAEfficacy(cri)
                    iteration = 1
                    pattern = "classwise_with_delta"
                    result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, device, pattern, seed)    
                    result = "{:.3f}".format(result)
                    res[cri] = result
                    acc_file.write(cri+"\t"+str(result)+"\t")
                acc_file.write("\n")
                acc_file.close()
                
            delta_cpu = delta.cpu()
            output_path = os.path.join(result_path_new,f'delta_epoch_{epoch}.pth')
            torch.save(delta_cpu, output_path)
                
    elif args.phase == "train" and args.num_indexes_to_replace is not None:
        result_path_new = os.path.join(args.result_path, exp_name_new)
        os.makedirs(result_path_new, exist_ok=True)
        criterion = nn.CrossEntropyLoss(reduction='sum')
    
        model = unlearn.load_original_checkpoint(model, device, 159, args)

        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    
        if args.init_delta == "random":
        #version 1: randomly initialize
            delta = torch.randn(unlearn_data_sets["retain"][0][0].shape).to(device)  
            delta_version = "from random"
        elif args.init_delta == "0":
        #version 2: start from zero, start from original input
            delta = torch.zeros_like(unlearn_data_sets["retain"][0][0]).to(device)  
            delta_version = "from 0"
        delta.requires_grad_(True)

        optimizer = torch.optim.SGD(
            [delta],
            args.unlearn_lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay,
        )
        scheduler = None
        if args.schedule==1:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        else:
            print("no scheduler")

        Tau = args.Tau
        Alpha = args.Alpha
        Beta = args.Beta
        Lambda = args.Lambda

        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d-%H-%M")
        name = exp_name_new

        for epoch in range(0, args.unlearn_epochs+1):

            current_time = datetime.now()
            current_time = current_time.strftime("%Y-%m-%d %H:%M")
            # print("epoch",epoch,"########################################","data-wise",args.percent,current_time,"Tau",Tau,"Alpha",Alpha,"Beta",Beta,"Lambda",Lambda,"delta_version",delta_version,"exp_name",exp_name_new,)

            from torch.utils.data import ConcatDataset
            mask = [0] * len(forget_set) + [1] * len(retain_set)
            combined_dataset = MaskedDataset(forget_set, retain_set, mask)
            combined_dataset_loader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
            
            
            all_loss = 0.0
            all_loss_f = 0.0
            all_loss_r = 0.0
            all_loss_norm = 0.0
            num_elements_equal_to_neg_tau = 0
            num_elements_retain = 0
        
            start_time = time.time()

            for i, (image, target, source) in enumerate(tqdm(combined_dataset_loader, desc='main_delta_2loss.py')): 

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
                l_dist = torch.norm(delta, p=2)/image.shape[0]
                
                loss = Alpha * l_r + Beta * l_f +  Lambda * l_dist
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                all_loss += loss.item()  
                all_loss_f += l_f.item()
                all_loss_r += l_r.item()
                all_loss_norm += l_dist.item()

            if args.schedule==1:
                print("scheduler")
                scheduler.step()
            else:
                print("no scheduler")
            
            delta_l2_norm = torch.norm(delta, p=2)

            loss_file = open(os.path.join(result_path_new,"loss.txt"),"a")
            # print("num_elements_equal_to_neg_tau/len(forget)",num_elements_equal_to_neg_tau,"/",len(forget_set))#被错误分类
            # print("num_elements_retain/len(retain)",num_elements_retain,"/",len(retain_set))##被正确分类
            # print('Epoch:[{0}]\t''Loss {loss:.4f}\t''all_loss_f {all_loss_f:.4f}\t''all_loss_r {all_loss_r:.4f}\t''delta_l2_norm {delta_l2_norm:.4f}\t'.format(epoch, loss=all_loss,all_loss_f = all_loss_f,all_loss_r = all_loss_r,delta_l2_norm=delta_l2_norm))
            loss_file.write(str("epoch")+str(epoch)+"\t"+str("all_loss")+"\t"+str(all_loss)+"\t"+str("all_loss_f")+"\t"+str(all_loss_f)+
                            "\t"+str("all_loss_r")+"\t"+str(all_loss_r)+"\t"+str("all_loss_norm")+"\t"+str(all_loss_norm)+"\t"+str("delta_l2_norm")+"\t"+str(delta_l2_norm)+"\t"+"\n")
            loss_file.close()
            
            
            retain_loader = DataLoader(
                        retain_for_test_set, batch_size=args.batch_size, num_workers=2, shuffle=False
                    )
            forget_loader = DataLoader(
                        forget_for_test_set, batch_size=args.batch_size, num_workers=2, shuffle=False
                    )
            val_loader = DataLoader(
                        val_set, batch_size=args.batch_size, num_workers=2, shuffle=False
                    )
            unlearn_data_loaders_acc = OrderedDict(
            retain=retain_loader, 
            forget=forget_loader, 
            val = val_loader
                    )
            
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
                
            modified_forget_set = ModifiedDataset(forget_for_test_set, delta)
            delta_forget_set_loader = torch.utils.data.DataLoader(modified_forget_set, batch_size=args.batch_size, shuffle=False)

            modified_retain_set = ModifiedDataset(retain_for_test_set, delta)
            delta_retain_set_loader = torch.utils.data.DataLoader(modified_retain_set, batch_size=args.batch_size,  shuffle=False)

            modified_val_set = ModifiedDataset(val_set, delta)
            delta_val_set_loader = torch.utils.data.DataLoader(modified_val_set, batch_size=args.batch_size, shuffle=False)
                    
            unlearn_data_loaders_mia = OrderedDict(
            retain=retain_loader, 
            forget=forget_loader, 
            val = val_loader,
            retain_delta = delta_retain_set_loader, 
            forget_delta = delta_forget_set_loader, 
            val_delta = delta_val_set_loader
            )
                    
            if epoch%10==0 and args.dataset=="cifar10":
                
                acc_file = open(os.path.join(result_path_new,"acc.txt"),"a")
                acc_file.write(str("epoch")+str(epoch)+"\t")
                for name, loader in unlearn_data_loaders_acc.items():
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    print("epoch",epoch,"start testing",name,current_time)
                    val_acc = delta_validate(loader, model, criterion, delta, args)
                    val_acc = "{:.3f}".format(val_acc)
                    res[name] = val_acc
                    acc_file.write(name+"\t"+str(val_acc)+"\t")
                    
                criterions = ["confidence"]
                for cri in criterions:
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    mia_efficacy = MIAEfficacy(cri)
                    iteration = 1 
                    pattern = "datawise_with_delta"
                    result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, device, pattern, seed)    
                    result = "{:.3f}".format(result)
                    res[cri] = result
                    acc_file.write(cri+"\t"+str(result)+"\t")
                acc_file.write("\n")
                acc_file.close()
                
                print("with delta",res)  
                    
            elif epoch%10==0 and args.dataset=="imagenet10":
                
                acc_file = open(os.path.join(result_path_new,"acc.txt"),"a")
                acc_file.write(str("epoch")+str(epoch)+"\t")
                for name, loader in unlearn_data_loaders_acc.items():
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    print("epoch",epoch,"start testing",name,current_time)
                    val_acc = delta_validate(loader, model, criterion, delta, args)
                    val_acc = "{:.3f}".format(val_acc)
                    res[name] = val_acc
                    acc_file.write(name+"\t"+str(val_acc)+"\t")
                    
                criterions = ["confidence"]
                for cri in criterions:
                    current_time = datetime.now()
                    current_time = current_time.strftime("%Y-%m-%d %H:%M")
                    mia_efficacy = MIAEfficacy(cri)
                    iteration = 1 
                    pattern = "datawise_with_delta"
                    result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, device, pattern, seed)    
                    result = "{:.3f}".format(result)
                    res[cri] = result
                    acc_file.write(cri+"\t"+str(result)+"\t")
                acc_file.write("\n")
                acc_file.close()
                
                print("yes delta",res)  

            delta_cpu = delta.cpu()
            output_path = os.path.join(result_path_new,f'delta_epoch_{epoch}.pth')
            torch.save(delta_cpu, output_path)
if __name__ == "__main__":
    main()
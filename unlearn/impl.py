import os
import sys
import time
from evaluation.mia import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

import pruner
import utils
from pruner import extract_mask, prune_model_custom, remove_prune
import datetime
# from datetime import datetime
sys.path.append(".")
from trainer import validate
import copy

def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()

def save_unlearn_checkpoint(model, evaluation_result, epoch, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    
    if args.unlearn == "retrain":
        utils.save_checkpoint(state, False, args.save_dir, args.unlearn,epoch)
        
        utils.save_checkpoint(
            evaluation_result,
            False,
            args.save_dir,
            args.unlearn,epoch,
            filename="eval_result.pth.tar",
        )
    else:
        utils.save_checkpoint(state, False, args.result_path, args.unlearn,epoch)
        
        utils.save_checkpoint(
            evaluation_result,
            False,
            args.result_path,
            args.unlearn,epoch,
            filename="eval_result.pth.tar",
        )
    
def save_unlearn_checkpoint_baseline(model, evaluation_result, epoch, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, args.result_path, args.unlearn,epoch)
    
    utils.save_checkpoint(
        evaluation_result,
        False,
        args.result_path,
        args.unlearn,epoch,
        filename="eval_result.pth.tar",
    )


def load_unlearn_checkpoint_saliency(model, device, args):
    # checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    checkpoint = utils.load_checkpoint(device, args.save_dir,"RL10_")
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    
    model.eval()

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result

def load_original_checkpoint(model, device,epoch, args):
    
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn+str(epoch)+"_")
    if checkpoint is None or checkpoint.get("state_dict") is None:
        print("model is None")
        return None

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    model.eval()
    
    return model

def load_unlearn_checkpoint(model, device,epoch, args):
    checkpoint = None
    print(args.save_dir, args.unlearn + str(epoch)+"_")
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn + str(epoch)+"_")

    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    model.eval()

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result

def load_unlearn_checkpoint_original(model, device, epoch, args):
    checkpoint = None
    checkpoint = utils.load_checkpoint(device, args.save_dir, "retrain"+str(epoch)+"_")

    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    model.eval()

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result

def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        model_t = None
        if args.unlearn == "SCRUB":
            model_t = copy.deepcopy(model)
            model_t.eval()
    
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        ##args.decreasing_lr: default="91,136",
        if args.unlearn == "retrain":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            ) 
        elif args.unlearn == "RL":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            ) 
        elif args.unlearn == "NegGrad_plus":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            ) 
        elif args.unlearn == "GA":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            ) 
        elif args.unlearn == "FT":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )
        elif args.unlearn == "IU":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )
        #SCRUB,
        elif args.unlearn == "SCRUB":
            scheduler=None
            
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                if scheduler is not None:
                    scheduler.step()
                # scheduler.step()
                
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            print("unlearn->impl.py->_iterative_unlearn_impl.")
            if args.unlearn == "SCRUB":
                train_acc = unlearn_iter_func(
                    data_loaders, model_t, model, criterion, optimizer, epoch, args, **kwargs
                )
            else:
                train_acc = unlearn_iter_func(
                    data_loaders, model, criterion, optimizer, epoch, args, **kwargs
                )
            
            if scheduler is not None:
                scheduler.step()
            # scheduler.step()
            #elapsed_time_mins = elapsed_time_secs / 60
            import unlearn
            #def save_unlearn_checkpoint(model, evaluation_result, args):
            unlearn.save_unlearn_checkpoint(model, None, epoch, args)
            
            unlearn_data_loaders = None
            # current_time = datetime.now()
            # current_time = current_time.strftime("%Y-%m-%d-%H-%M")
            # print("epoch",epoch,current_time)
            if args.class_to_replace is not None and args.num_indexes_to_replace is None:
                print("class-wise")
                pattern = "classwise"
                unlearn_data_loaders_acc = OrderedDict(
                    retain=data_loaders["retain_for_test"], 
                    forget=data_loaders['forget_for_test'],
                    val_retain=data_loaders['val_retain'], 
                    val_forget=data_loaders['val_forget']
                )
                unlearn_data_loaders_mia = OrderedDict(
                    retain=data_loaders["retain"], 
                    forget=data_loaders['forget'],
                    val_retain=data_loaders['val_retain'], 
                    val_forget=data_loaders['val_forget']
                )
            elif args.class_to_replace is None and args.num_indexes_to_replace is not None:
                print("data-wise")
                pattern = "datawise"
                unlearn_data_loaders_acc = OrderedDict(
                    retain=data_loaders["retain_for_test"], 
                    forget=data_loaders['forget_for_test'],
                    val=data_loaders['val'], 
                )
                unlearn_data_loaders_mia = OrderedDict(
                    retain=data_loaders["retain"], 
                    forget=data_loaders['forget'],
                    val=data_loaders['val'], 
                )
            # if args.unlearn == "FT" and epoch%1==0:
            if args.unlearn == "FT" or args.unlearn == "GA" or args.unlearn == "RL" or args.unlearn == "IU" or args.unlearn=="NegGrad_plus" or args.unlearn=="SCRUB":
                print("Start test_classwise.","epoch",epoch)
                criterion = nn.CrossEntropyLoss()
                evaluation_result = None
                val_result = open(os.path.join(args.result_path,"acc.txt"),"a")
                val_result.write(str(epoch)+"\t")
                for name, loader in unlearn_data_loaders_acc.items():
            # utils.dataset_convert_to_test(loader.dataset,args)
                    val_acc = validate(loader, model, criterion, args)
                    val_acc = "{:.3f}".format(val_acc)
                    print(f"epoch: {epoch},{name},acc: {val_acc}")
                    val_result.write(name+"\t"+str(val_acc)+"\t")
                # val_result.write("\n")
                # val_result.close()
                if epoch%1==0: 
                    criterions = ["confidence"]
                    for cri in criterions:
                        mia_efficacy = MIAEfficacy(cri)
                    # Call the evaluate method
                        iteration = 1
                        result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, torch.device(f"cuda:{int(args.gpu)}"), pattern, args.train_seed)
                        result = "{:.3f}".format(result)
                        print(f"epoch: {epoch},mia: {result}")
                        val_result.write(cri+"\t"+result+"\t")            
                    val_result.write("\n")
                    val_result.close()
                else:
                    val_result.write("\n")
                    val_result.close()
                    
            if args.unlearn == "retrain" and epoch%10==0:
                print("Start test.","epoch",epoch)
                criterion = nn.CrossEntropyLoss()
                evaluation_result = None
                val_result = open(os.path.join(args.result_path,"acc.txt"),"a")
                val_result.write(str(epoch)+"\t")
                for name, loader in unlearn_data_loaders_acc.items():
            # utils.dataset_convert_to_test(loader.dataset,args)
                    val_acc = validate(loader, model, criterion, args)
                    val_acc = "{:.3f}".format(val_acc)
                    print(f"epoch: {epoch},{name},acc: {val_acc}")
                    val_result.write(name+"\t"+str(val_acc)+"\t")
                # val_result.write("\n")
                # val_result.close()
                if epoch%1==0: 
                    # current_time = datetime.datetime.now()
                    # formatted_time = current_time.strftime("%Y-%m-%d-%H-%M")
                    # print(formatted_time)
                    criterions = ["confidence"]
                    for cri in criterions:
                        mia_efficacy = MIAEfficacy(cri)
                    # Call the evaluate method
                        iteration = 1
                        result = mia_efficacy.evaluate(model, unlearn_data_loaders_mia, iteration, torch.device(f"cuda:{int(args.gpu)}"), pattern, args.train_seed)
                        result = "{:.3f}".format(result)
                        print(f"epoch: {epoch},mia: {result}")
                        val_result.write(cri+"\t"+result+"\t")            
                    val_result.write("\n")
                    val_result.close()
                    
                    # current_time = datetime.datetime.now()
                    # formatted_time = current_time.strftime("%Y-%m-%d-%H-%M")
                    # print(formatted_time)
        
                else:
                    val_result.write("\n")
                    val_result.close()
            
                
    return _wrapped

def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    print("def iterative_unlearn(func):")
    return _iterative_unlearn_impl(func)

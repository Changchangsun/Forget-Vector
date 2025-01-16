"""
    setup model and datasets
"""


import copy
import os
import random

# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
import sys
import time

import numpy as np
import torch
from torchvision import transforms

from dataset import *
from dataset import TinyImageNet
from delta_imagenet_path import prepare_data
from models import *
from timm import create_model

__all__ = [
    "setup_model_dataset",
    "AverageMeter",
    "warmup_lr",
    "save_checkpoint",
    "setup_seed",
    "accuracy",
]


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr

#    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
def save_checkpoint(
    state, is_SA_best, save_path, pruning, epoch,filename="checkpoint.pth.tar"
):
    filepath = os.path.join(save_path, str(pruning) +str(epoch)+"_"+ filename)
    torch.save(state, filepath)
    if is_SA_best:#False
        shutil.copyfile(
            filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar")
        )

#    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dataset_convert_to_train(dataset):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = train_transform
    dataset.train = False


def dataset_convert_to_test(dataset, args=None):
    if args.dataset == "TinyImagenet":
        test_transform = transforms.Compose([])
    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False


def setup_model_dataset(args):
    if args.dataset == "cifar10":
    
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        # train_loader, val_loader, test_loader
        train_full_loader, val_full_loader, test_full_loader = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )
        marked_loader, _, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace,
            num_indexes_to_replace=args.num_indexes_to_replace,
            indexes_to_replace=args.indexes_to_replace,
            seed=args.seed,
            only_mark=True,
            shuffle=True,
            no_aug=args.no_aug, #False
        )

        if args.train_seed is None:#default=1,
            args.train_seed = args.seed
        setup_seed(args.train_seed)

        if args.imagenet_arch:#
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        elif args.arch == "swin_t":#args.arch:default="resnet18", help="model architecture"
            model = swin_t(
                window_size=4, num_classes=10, downscaling_factors=(2, 2, 2, 1)
            )
        else:#resnet18
            model = model_dict[args.arch](num_classes=classes)#resnet18(num_classes=10)

        setup_seed(args.train_seed)

        model.normalize = normalization
        # print(model)
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == "cifar10_scc":
        setup_seed(args.train_seed)##default = 1

        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        model = None
        if args.arch == "resnet18":
            model = model_dict[args.arch](num_classes=classes,imagenet=False)#resnet18(num_classes=10)
            print("3 Finish select the resnet18 model.")
        elif args.arch == "vgg16_bn":
            model = model_dict[args.arch](num_classes=classes)#resnet18(num_classes=10)
            print("3 Finish select the vgg16_bn model.")
            
        model.normalize = normalization
        
        if args.class_to_replace is None and args.num_indexes_to_replace is None:###original model
            print("#####original model")
            sets = prepare_data(dataset="cifar10_scc", batch_size=args.batch_size, class_to_replace = args.class_to_replace, indexes_to_replace = args.indexes_to_replace, 
                                   seed=args.train_seed, single = args.single, adv=args.adv, cor=args.cor, cor_type=args.cor_type, level=args.level, 
                                   phase=args.phase, data_path=args.data, arch=args.arch, percent = args.percent)
            train_set, train_set_for_test, val_set = sets["train"], sets["train_for_test"], sets["val"]
            return model, train_set, train_set_for_test, val_set
        
        elif args.class_to_replace is not None and args.num_indexes_to_replace is None:#unlearning class-wise
            ##class-wise
            print("###class wise")
            sets = prepare_data(dataset="cifar10_scc", batch_size=args.batch_size,class_to_replace = args.class_to_replace, indexes_to_replace = args.indexes_to_replace, adv =args.adv,cor = args.cor,cor_type = args.cor_type,level = args.level,phase = args.phase,data_path = args.data)            
            retain_set, forget_set, retain_set_for_test, forget_set_for_test, val_set, val_set_retain, val_set_forget, retain_set_adv, forget_set_adv, val_set_adv, val_retain_set_adv, val_forget_set_adv = sets["retain"], sets["forget"], sets["retain_for_test"], sets["forget_for_test"], sets['val'], sets['val_retain'], sets['val_forget'], sets["retain_adv"], sets["forget_adv"], sets["val_adv"], sets['val_retain_adv'],sets['val_forget_adv']
            return model,retain_set, forget_set, retain_set_for_test, forget_set_for_test, val_set, val_set_retain, val_set_forget, retain_set_adv, forget_set_adv, val_set_adv, val_retain_set_adv, val_forget_set_adv 
        
        elif args.class_to_replace is None and args.num_indexes_to_replace is not None:#unlearning class-wise
            print("###data wise")
            sets = prepare_data(dataset="cifar10_scc", batch_size=args.batch_size,class_to_replace = args.class_to_replace, indexes_to_replace = args.indexes_to_replace,num_indexes_to_replace=args.num_indexes_to_replace,seed=args.seed,adv =args.adv,cor = args.cor,cor_type = args.cor_type,level = args.level,phase = args.phase,data_path = args.data, percent = args.percent)
            print("###loaders end")
            retain_set, forget_set, retain_set_for_test, forget_set_for_test, val_set, retain_set_adv, forget_set_adv, val_set_adv = sets["retain"], sets["forget"], sets['retain_for_test'], sets['forget_for_test'], sets["val"], sets["retain_adv"], sets["forget_adv"], sets["val_adv"]
            return model, retain_set, forget_set, retain_set_for_test, forget_set_for_test, val_set, retain_set_adv, forget_set_adv, val_set_adv
    elif args.dataset == "imagenet10":
        classes = 10
        setup_seed(args.train_seed)##default = 1

        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        model = None
        if args.arch == "resnet18":
            model = model_dict[args.arch](num_classes=classes,imagenet=True)#resnet18(num_classes=10)
            print("3 Finish select resnet18 model.")
        elif args.arch == "vgg16_bn":
            ####
            model = model_dict[args.arch](num_classes=classes)#resnet18(num_classes=10)
            print("3 Finish select vgg16_bn model.")
        elif args.arch == "vit":
            model = create_model('vit_base_patch16_224', pretrained=True)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, classes)  # CIFAR-10 有 10 个分类
            print("vit model arch")
            print(model)

        model.normalize = normalization
        
        if args.class_to_replace is None and args.num_indexes_to_replace is None:###original model
            print("###original model")
            sets = prepare_data(dataset="imagenet10", batch_size=args.batch_size,class_to_replace = args.class_to_replace, indexes_to_replace = args.indexes_to_replace, single = args.single, phase = args.phase,data_path = args.data)
            # train_loader, val_loader,test_loader = loaders["train"], loaders["val"],loaders["test"]
            train_set, train_set_for_test, val_set = sets["train"], sets["train_for_test"], sets["val"]
            return model, train_set, train_set_for_test, val_set
            
        elif args.class_to_replace is not None and args.num_indexes_to_replace is None:#unlearning class-wise
            print("###class wise")
            sets = prepare_data(dataset="imagenet10", batch_size=args.batch_size,class_to_replace = args.class_to_replace, indexes_to_replace = args.indexes_to_replace, adv =args.adv,cor = args.cor,cor_type = args.cor_type,level = args.level,phase = args.phase,data_path = args.data)
            retain_set, forget_set, retain_set_for_test, forget_set_for_test, val_set, val_set_retain, val_set_forget, retain_set_adv, forget_set_adv, val_set_adv, val_retain_set_adv, val_forget_set_adv = sets["retain"], sets["forget"], sets["retain_for_test"], sets["forget_for_test"], sets['val'], sets['val_retain'], sets['val_forget'], sets["retain_adv"], sets["forget_adv"], sets["val_adv"], sets['val_retain_adv'],sets['val_forget_adv']
            return model,retain_set, forget_set, retain_set_for_test, forget_set_for_test, val_set, val_set_retain, val_set_forget, retain_set_adv, forget_set_adv, val_set_adv, val_retain_set_adv, val_forget_set_adv 
        
        elif args.class_to_replace is None and args.num_indexes_to_replace is not None:#unlearning class-wise
            print("###data wise")
            sets = prepare_data(dataset="imagenet10", batch_size=args.batch_size,class_to_replace = args.class_to_replace, indexes_to_replace = args.indexes_to_replace,num_indexes_to_replace=args.num_indexes_to_replace,seed=args.seed,adv =args.adv,cor = args.cor,cor_type = args.cor_type,level = args.level,phase = args.phase,data_path = args.data)
            retain_set, forget_set, retain_set_for_test, forget_set_for_test, val_set, retain_set_adv, forget_set_adv, val_set_adv = sets["retain"], sets["forget"], sets['retain_for_test'], sets['forget_for_test'], sets["val"], sets["retain_adv"], sets["forget_adv"], sets["val_adv"]
            return model, retain_set, forget_set, retain_set_for_test, forget_set_for_test, val_set, retain_set_adv, forget_set_adv, val_set_adv                    
        
    elif args.dataset == "cifar10_scc_transfer2":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        model = None
        if args.arch == "resnet18":
            model = model_dict[args.arch](num_classes=classes,imagenet=False)#resnet18(num_classes=10)
            print("3 Finish select the resnet18 model.")
        elif args.arch == "vgg16_bn":
            model = model_dict[args.arch](num_classes=classes)#resnet18(num_classes=10)
            print("3 Finish select the vgg16_bn model.")

        model.normalize = normalization
        
        if args.multi_classes_to_replace is not None:#unlearning class-wise
            print("###class wise")
            sets = prepare_data(dataset="cifar10_scc_transfer2", batch_size=args.batch_size,class_to_replace = args.class_to_replace, indexes_to_replace = args.indexes_to_replace, adv =args.adv,cor = args.cor,cor_type = args.cor_type,level = args.level,phase = args.phase,data_path = args.data, multi_classes_to_replace = args.multi_classes_to_replace,num_classes = classes)
            retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, val_retain_set, val_forget_set = sets["retain"], sets["forget"], sets["retain_for_test"], sets["forget_for_test"], sets['val'], sets['val_retain'], sets['val_forget']
            return model, retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, val_retain_set, val_forget_set
    
        
    elif args.dataset == "imagenet10_transfer2":
        classes = 10
        setup_seed(args.train_seed)##default = 1

        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        

        model = None
        if args.arch == "resnet18":
            model = model_dict[args.arch](num_classes=classes,imagenet=True)#resnet18(num_classes=10)
            print("3 Finish select resnet18 model.")
        elif args.arch == "vgg16_bn":
            model = model_dict[args.arch](num_classes=classes)#resnet18(num_classes=10)
            print("3 Finish select vgg16_bn model.")
        elif args.arch == "vit":
            model = create_model('vit_base_patch16_224', pretrained=True)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, classes)  # CIFAR-10 有 10 个分类
            print("vit model arch")
            print(model)

        model.normalize = normalization
        
        if args.multi_classes_to_replace is not None:#unlearning class-wise
            print("###class wise")
            sets = prepare_data(dataset="imagenet10_transfer2", batch_size=args.batch_size,class_to_replace = args.class_to_replace, indexes_to_replace = args.indexes_to_replace, adv =args.adv,cor = args.cor,cor_type = args.cor_type,level = args.level,phase = args.phase,data_path = args.data, multi_classes_to_replace = args.multi_classes_to_replace,num_classes = classes)
            retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, val_retain_set, val_forget_set = sets["retain"], sets["forget"], sets["retain_for_test"], sets["forget_for_test"], sets['val'], sets['val_retain'], sets['val_forget']
            return model, retain_set, forget_set, retain_for_test_set, forget_for_test_set, val_set, val_retain_set, val_forget_set
    else:
        raise ValueError("Dataset not supprot yet !")
    # import pdb;pdb.set_trace()

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    print(model)

    return model, train_set_loader, val_loader, test_loader


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if len(commands) == 0:
        return
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)

    fout = open("stop_{}.sh".format(dir), "w")
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, "w")
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)


def get_loader_from_dataset(dataset, batch_size, seed=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=shuffle
    )


def get_unlearn_loader(marked_loader, args):
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = get_loader_from_dataset(
        forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = get_loader_from_dataset(
        retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    print("datasets length: ", len(forget_dataset), len(retain_dataset))
    return forget_loader, retain_loader


def get_poisoned_loader(poison_loader, unpoison_loader, test_loader, poison_func, args):
    poison_dataset = copy.deepcopy(poison_loader.dataset)
    poison_test_dataset = copy.deepcopy(test_loader.dataset)

    poison_dataset.data, poison_dataset.targets = poison_func(
        poison_dataset.data, poison_dataset.targets
    )
    poison_test_dataset.data, poison_test_dataset.targets = poison_func(
        poison_test_dataset.data, poison_test_dataset.targets
    )

    full_dataset = torch.utils.data.ConcatDataset(
        [unpoison_loader.dataset, poison_dataset]
    )

    poisoned_loader = get_loader_from_dataset(
        poison_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    poisoned_full_loader = get_loader_from_dataset(
        full_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    poisoned_test_loader = get_loader_from_dataset(
        poison_test_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )

    return poisoned_loader, unpoison_loader, poisoned_full_loader, poisoned_test_loader

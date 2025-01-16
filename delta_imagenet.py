import os
import sys

import torch
import torchvision
from datasets.load import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from torchvision.datasets import ImageFolder
import os
import copy

class ImageFolderWithPaths(ImageFolder):
    # Override the __getitem__ method to also return the image file path
    def __getitem__(self, index):
        # This is the standard implementation of `ImageFolder`'s __getitem__ method
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path

def prepare_data(
    dataset,
    batch_size=512,
    shuffle=True,
    class_to_replace=None,
    indexes_to_replace=None,
    num_indexes_to_replace=None,
    train_subset_indices=None,
    val_subset_indices=None,
    only_mark: bool = True, #True
    seed: int=0,
    adv: str=None,
    cor: str=None,
    cor_type: str=None,
    level: str=None,
    single: int=0,
    phase: str="train",
    unlearn: str="retrain",
    data_path: str=None,
    percent: int=10,
    arch: str="resnet18",
    multi_classes_to_replace: int=0,
    num_classes: int=10,
):
    if dataset == "imagenet10":
        train_transform = transforms.Compose([
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.RandomResizedCrop((224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
    )

        validation_transform = transforms.Compose([
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                ])
        
        train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        
        train_set_for_test = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=validation_transform)

        val_set = datasets.ImageFolder(os.path.join(data_path, 'validation'), transform=validation_transform)
        
        train_set_adv = None
        val_set_adv = None
        
        if adv is not None and cor is None:
            adv_path = adv
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train_same_folder"), transform=validation_transform)
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"validation_same_folder"), transform=validation_transform)
        elif adv is None and cor is not None:
            adv_path = cor
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train",cor_type,level), transform=validation_transform)
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"validation",cor_type,level), transform=validation_transform)
        else:
            print("Wrong perturbation datasets.")
                    
        if class_to_replace is not None and num_indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        elif class_to_replace is None and num_indexes_to_replace is None:
            sets = {
                "train": train_set,
                "train_for_test":train_set_for_test,
                "val":val_set,
            }
            return sets
    
        elif class_to_replace is not None and num_indexes_to_replace is None:
            forget_indices = np.flatnonzero(np.array(train_set.targets) == class_to_replace)
            retain_indices = np.flatnonzero(np.array(train_set.targets) != class_to_replace)
            
            forget_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) == class_to_replace)
            retain_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) != class_to_replace)
        
            forget_indices_val = np.flatnonzero(np.array(val_set.targets) == class_to_replace)
            retain_indices_val = np.flatnonzero(np.array(val_set.targets) != class_to_replace)
            
            forget_set = Subset(train_set, forget_indices)
            retain_set = Subset(train_set, retain_indices)
            
            forget_set_for_test = Subset(train_set_for_test, forget_indices_for_test) 
            retain_set_for_test = Subset(train_set_for_test, retain_indices_for_test)
            
            forget_set_val = Subset(val_set, forget_indices_val)
            retain_set_val = Subset(val_set, retain_indices_val)
                
            retain_set_adv = None
            forget_set_adv = None
            retain_set_val_adv = None
            forget_set_val_adv = None

            retain_set_adv = Subset(train_set_adv, retain_indices)
            forget_set_adv = Subset(train_set_adv, forget_indices)
            retain_set_val_adv = Subset(val_set_adv, retain_indices_val)
            forget_set_val_adv = Subset(val_set_adv, forget_indices_val)
                
            print(f"retain set size: {len(retain_set)}")
            print(f"forget set size: {len(forget_set)}")
            print(f"retain_for_test set size: {len(retain_set_for_test)}")
            print(f"forget_for_test set size: {len(forget_set_for_test)}")
            print(f"val set size: {len(val_set)}")
            print(f"val_retain set size: {len(retain_set_val)}")
            print(f"val_forget set size: {len(forget_set_val)}")
            
            print(f"retain_adv set size: {len(retain_set_adv)}")
            print(f"forget_adv set size: {len(forget_set_adv)}")
            print(f"val_adv set size: {len(val_set_adv)}")
            print(f"val_retain_adv set size: {len(retain_set_val_adv)}")
            print(f"val_forget_adv set size: {len(forget_set_val_adv)}")
            
            sets = {
                    "retain": retain_set,
                    "forget":forget_set,
                    "retain_for_test":retain_set_for_test,
                    "forget_for_test":forget_set_for_test,
                    "val": val_set,
                    "val_retain": retain_set_val,
                    "val_forget": forget_set_val,
                    "retain_adv": retain_set_adv,
                    "forget_adv": forget_set_adv,
                    "val_adv": val_set_adv,    
                    "val_retain_adv": retain_set_val_adv,
                    "val_forget_adv":forget_set_val_adv,
                    }
            return sets
        
        elif class_to_replace is None and num_indexes_to_replace is not None:
            len_train = len(train_set)
            assert len_train == 13000, f"The number for data-wise of imagenet10 setting is wrong."

            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)
            forget_indices = indexes
            # print("forget_indices",forget_indices)
            # print(int(len_train*0.1),"forget length")
        
            all_index = np.arange(len_train)
            mask = ~np.isin(all_index, forget_indices)
            retain_indices = all_index[mask]

            forget_set = Subset(train_set, forget_indices)
            retain_set = Subset(train_set, retain_indices)

            forget_set_for_test = Subset(train_set_for_test, forget_indices)
            retain_set_for_test = Subset(train_set_for_test, retain_indices)
        
            retain_set_adv = Subset(train_set_adv, retain_indices)
            forget_set_adv = Subset(train_set_adv, forget_indices)
            
            print(f"retain set size: {len(retain_set)}")
            print(f"forget set size: {len(forget_set)}")
            print(f"retain_for_test set size: {len(retain_set_for_test)}")
            print(f"forget_for_test set size: {len(forget_set_for_test)}")
            print(f"val set size: {len(val_set)}")
            print(f"retain_adv set size: {len(retain_set_adv)}")
            print(f"forget_adv set size: {len(forget_set_adv)}")
            print(f"val_adv set size: {len(val_set_adv)}")
            
            sets = {
                    "retain": retain_set,
                    "forget":forget_set,
                    "retain_for_test":retain_set_for_test,
                    "forget_for_test":forget_set_for_test,
                    "val": val_set,
                    "retain_adv": retain_set_adv,
                    "forget_adv": forget_set_adv,
                    "val_adv": val_set_adv,    
                    }
            return sets
        
    elif dataset == "cifar10":
        
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
        )        
        
        train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        train_set_for_test = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=test_transform)
        val_set = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=test_transform)

        
        train_set_adv = None
        if adv is not None and cor is None:
            adv_path = adv
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train_same_folder"), transform=test_transform)
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"test_same_folder"), transform=test_transform)
        elif  adv is None and cor is not None:
            adv_path = cor
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train",cor_type,level), transform=test_transform)
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"test",cor_type,level), transform=test_transform)
        else:
            print("Wrong perturbation datasets.")
        
        if class_to_replace is not None and num_indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        elif class_to_replace is None and num_indexes_to_replace is None:
            sets = {
                    "train": train_set,
                    "train_for_test": train_set_for_test,
                    "val":val_set,
                    }
            return sets
    
        elif class_to_replace is not None and num_indexes_to_replace is None:
            forget_indices = np.flatnonzero(np.array(train_set.targets) == class_to_replace)
            retain_indices = np.flatnonzero(np.array(train_set.targets) != class_to_replace)
            
            forget_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) == class_to_replace)
            retain_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) != class_to_replace)
            
            forget_indices_val = np.flatnonzero(np.array(val_set.targets) == class_to_replace)
            retain_indices_val = np.flatnonzero(np.array(val_set.targets) != class_to_replace)
            
            retain_set = Subset(train_set, retain_indices)
            forget_set = Subset(train_set, forget_indices)   
             
            forget_set_for_test = Subset(train_set_for_test, forget_indices_for_test) 
            retain_set_for_test = Subset(train_set_for_test, retain_indices_for_test)
            
            retain_set_val = Subset(val_set, retain_indices_val)
            forget_set_val = Subset(val_set, forget_indices_val)
            
            retain_set_adv = None
            forget_set_adv = None
            retain_set_val_adv = None
            forget_set_val_adv = None
            
            retain_set_adv = Subset(train_set_adv, retain_indices)
            forget_set_adv = Subset(train_set_adv, forget_indices)
            retain_set_val_adv = Subset(val_set_adv, retain_indices_val)
            forget_set_val_adv = Subset(val_set_adv, forget_indices_val)
            
            print(f"retain set size: {len(retain_set)}")
            print(f"forget set size: {len(forget_set)}")
            print(f"retain_for_test set size: {len(retain_set_for_test)}")
            print(f"forget_for_test set size: {len(forget_set_for_test)}")
            print(f"val set size: {len(val_set)}")
            print(f"val_retain set size: {len(retain_set_val)}")
            print(f"val_forget set size: {len(forget_set_val)}")
            
            print(f"retain_adv set size: {len(retain_set_adv)}")
            print(f"forget_adv set size: {len(forget_set_adv)}")
            print(f"val_adv set size: {len(val_set_adv)}")
            print(f"val_retain_adv set size: {len(retain_set_val_adv)}")
            print(f"val_forget_adv set size: {len(forget_set_val_adv)}")
                
            sets = {
                    "retain": retain_set,
                    "forget":forget_set,
                    "retain_for_test":retain_set_for_test,
                    "forget_for_test":forget_set_for_test,
                    "val": val_set,
                    "val_retain": retain_set_val,
                    "val_forget": forget_set_val,
                    "retain_adv": retain_set_adv,
                    "forget_adv": forget_set_adv,
                    "val_adv": val_set_adv,    
                    "val_retain_adv": retain_set_val_adv,
                    "val_forget_adv":forget_set_val_adv,
                    }
            return sets
        
        elif class_to_replace is None and num_indexes_to_replace is not None:
            len_train = len(train_set)
            assert len_train == 50000, f"The number for data-wise of cifar10 setting is wrong."
            
            indexes = None
            forget_indices = None
            retain_indices = None
            
            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)#replace=False：表示抽样时不允许重复
            forget_indices = indexes
            
            all_index = np.arange(len_train)
            mask = ~np.isin(all_index, forget_indices)
            retain_indices = all_index[mask]
            
            retain_set = Subset(train_set, retain_indices)
            forget_set = Subset(train_set, forget_indices)
            
            forget_set_for_test = Subset(train_set_for_test, forget_indices)
            retain_set_for_test = Subset(train_set_for_test, retain_indices)
            
            retain_set_adv = Subset(train_set_adv, retain_indices)
            forget_set_adv = Subset(train_set_adv, forget_indices)
            
            print(f"retain set size: {len(retain_set)}")
            print(f"forget set size: {len(forget_set)}")
            print(f"retain_for_test set size: {len(retain_set_for_test)}")
            print(f"forget_for_test set size: {len(forget_set_for_test)}")
            print(f"val set size: {len(val_set)}")
            print(f"retain_adv set size: {len(retain_set_adv)}")
            print(f"forget_adv set size: {len(forget_set_adv)}")
            print(f"val_adv set size: {len(val_set_adv)}")
            
            sets = {
                    "retain": retain_set,
                    "forget":forget_set,
                    "retain_for_test":retain_set_for_test,
                    "forget_for_test":forget_set_for_test,
                    "val": val_set,
                    "retain_adv": retain_set_adv,
                    "forget_adv": forget_set_adv,
                    "val_adv": val_set_adv,    
                    }
            return sets
    else:
        raise NotImplementedError


def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    ys = {}
    ys["train"] = []
    ys["val"] = []
    loaders = prepare_data(dataset="imagenet", batch_size=1, shuffle=False)
    for data in tqdm(loaders["val"], ncols=100):
        x, y = get_x_y_from_data_dict(data, "cpu")
        ys["val"].append(y.item())
    for data in tqdm(loaders["train"], ncols=100):
        x, y = get_x_y_from_data_dict(data, "cpu")
        ys["train"].append(y.item())
    ys["train"] = torch.Tensor(ys["train"]).long()
    ys["val"] = torch.Tensor(ys["val"]).long()
    torch.save(ys["train"], "train_ys.pth")
    torch.save(ys["val"], "val_ys.pth")

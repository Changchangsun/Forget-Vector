import os
import sys

import torch
import torchvision
from datasets.load import load_dataset
from torch.utils.data import DataLoader, Subset
# import datasets
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import numpy as np
# sys.path.append(".")
# from cfg import *
from tqdm import tqdm

from torchvision.datasets import ImageFolder
import os
import copy
class ImageFolderWithPaths(ImageFolder):
    # Override the __getitem__ method to also return the image file path
    def __getitem__(self, index):
        # This is the standard implementation of `ImageFolder`'s __getitem__ method
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Append the file path to the original tuple
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path

def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
            dataset._labels[indexes] = dataset._labels[new_indexes]
    else:

        dataset.targets = np.array(dataset.targets)
        dataset.targets[indexes] = -dataset.targets[indexes] - 1
        dataset.targets = dataset.targets.tolist()


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = True,
):
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    else:
        try:
            indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
            # print("replace_indexes#####",indexes)
        except:
            try:
                indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
            except:
                indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    print("replace_indexes directly##")
    replace_indexes(dataset, indexes, seed, only_mark)


def prepare_data(
    dataset,
    module = None, 
    multi_classes_to_replace = None,
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
    data_path="/egr/research-optml/sunchan5/MU/Unlearn-Sparse/datasets/ImageNet10",
    percent: int=10,
    arch: str="resnet18",
    num_classes: int=10,
):
    path = os.path.join(data_path, "huggingface")
    if dataset == "imagenet":
        train_set = load_dataset(
            "imagenet-1k", use_auth_token=True, split="train", cache_dir=path
        )
        validation_set = load_dataset(
            "imagenet-1k", use_auth_token=True, split="validation", cache_dir=path
        )

        def train_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.RandomResizedCrop((224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples
    elif dataset == "imagenet10":
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
        
                
        #train set 1
        train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        
        train_set_for_test = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=validation_transform)

        #val set 2
        val_set = datasets.ImageFolder(os.path.join(data_path, 'validation'), transform=validation_transform)
        
        
        train_set_adv = None
        val_set_adv = None
        
        if adv is not None:
            adv_path = adv
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train_same_folder"), transform=validation_transform)
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"validation_same_folder"), transform=validation_transform)
        elif cor is not None:
            adv_path = cor
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train",cor_type,level), transform=validation_transform)
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"validation",cor_type,level), transform=validation_transform)
            #
        else:
        if multi_classes_to_replace is not None and num_indexes_to_replace is not None:
            train_indexes = []
            for index in multi_classes_to_replace:
                train_indexes+=list(range(index*1300,(index+1)*1300))
            train_set_sub = Subset(train_set, train_indexes)
            train_set_for_test_sub = Subset(train_set_for_test,train_indexes)
            
            train_set_adv_sub = Subset(train_set_adv, train_indexes)
            val_indexes = []
            for index in multi_classes_to_replace:
                val_indexes+=list(np.flatnonzero(np.array(val_set.targets) == index))
            
            val_set_sub = Subset(val_set, val_indexes)
            val_set_adv_sub = Subset(val_set_adv,val_indexes)
            len_train = len(train_set_sub)
            
            indexes = None
            forget_indices = None
            retain_indices = None
            
            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)#
            forget_indices = indexes
            print("forget_indices",forget_indices)
            
            all_index = np.arange(len_train)
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
            retain_indices = all_index[mask]
            
            retain_set = Subset(train_set_sub, retain_indices)
            forget_set = Subset(train_set_sub, forget_indices)
            
            forget_set_for_test = Subset(train_set_for_test_sub, forget_indices)
            retain_set_for_test = Subset(train_set_for_test_sub, retain_indices)
            
            retain_set_adv = Subset(train_set_adv_sub, retain_indices)
            forget_set_adv = Subset(train_set_adv_sub, forget_indices)
            
            val_set = val_set_sub
            val_set_adv = val_set_adv_sub
            
            
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

                    
        elif class_to_replace is not None and num_indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        elif class_to_replace is None and num_indexes_to_replace is None:
            print("original")   

            sets = {
                "train": train_set,
                "train_for_test":train_set_for_test,
                "val":val_set,
                # "test":test_loader,
            }
            return sets
    
        elif class_to_replace is not None and num_indexes_to_replace is None:
            #class-wise
            print("class-wise")
            
            # new_train_set = copy.deepcopy(train_set)
        
            forget_indices = np.flatnonzero(np.array(train_set.targets) == class_to_replace)
            print("forget_indices",forget_indices)
            retain_indices = np.flatnonzero(np.array(train_set.targets) != class_to_replace)
            
            forget_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) == class_to_replace)
            print("forget_indices_for_test",forget_indices_for_test)
            retain_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) != class_to_replace)
            
            # assert forget_indices==forget_indices_for_test, f"Wrong1."
            # assert retain_indices==retain_indices_for_test, f"Wrong2."
            
            #validation set
            forget_indices_val = np.flatnonzero(np.array(val_set.targets) == class_to_replace)
            print("forget_indices_val",forget_indices_val)
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
            print("data-wise")
            len_train = len(train_set)
            assert len_train == 13000, f"The number for data-wise of imagenet10 setting is wrong."

            # new_train_set = copy.deepcopy(train_set)
            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            print("forget rate: #####################", percent/100)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)#
            forget_indices = indexes
            print("forget_indices",forget_indices)
            print(int(len_train*0.1),"forget length")
        
            all_index = np.arange(len_train)
            # print("all_index",all_index[0])
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
            retain_indices = all_index[mask]
            print("retain_indices",retain_indices)

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
        
    elif dataset == "cifar10_scc":
        if arch=="vit":
            train_transform = transforms.Compose(
                [
                transforms.Resize((224, 224)),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
            )
        else:
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
        # train_loader_adv = None
        if adv is not None:
            adv_path = adv
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train_same_folder"), transform=test_transform)
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"test_same_folder"), transform=test_transform)
        elif cor is not None:
            adv_path = cor
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train",cor_type,level), transform=test_transform)
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"test",cor_type,level), transform=test_transform)
        else:
            print("Wrong")
        
        if multi_classes_to_replace is not None and num_indexes_to_replace is not None:
            print("data-wise")
            print(len(val_set))
            # assert len_train == 50000, f"The number for data-wise of cifar10 setting is wrong."
            train_indexes = []
            for index in multi_classes_to_replace:
                train_indexes+=list(range(index*5000,(index+1)*5000))
            # print(train_indexes)
            train_set_sub = Subset(train_set, train_indexes)
            train_set_for_test_sub = Subset(train_set_for_test,train_indexes)
            
            train_set_adv_sub = Subset(train_set_adv, train_indexes)
            
            val_indexes = []
            for index in multi_classes_to_replace:
                val_indexes+=list(np.flatnonzero(np.array(val_set.targets) == index))
            
            val_set_sub = Subset(val_set, val_indexes)
            val_set_adv_sub = Subset(val_set_adv,val_indexes)
            len_train = len(train_set_sub)
            
            indexes = None
            forget_indices = None
            retain_indices = None
            
            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)
            forget_indices = indexes
            print("forget_indices",forget_indices)
            
            all_index = np.arange(len_train)
            # print("all_index",all_index[0])
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
            retain_indices = all_index[mask]
            
            retain_set = Subset(train_set_sub, retain_indices)
            forget_set = Subset(train_set_sub, forget_indices)
            
            forget_set_for_test = Subset(train_set_for_test_sub, forget_indices)
            retain_set_for_test = Subset(train_set_for_test_sub, retain_indices)
            
            retain_set_adv = Subset(train_set_adv_sub, retain_indices)
            forget_set_adv = Subset(train_set_adv_sub, forget_indices)
            
            val_set = val_set_sub
            val_set_adv = val_set_adv_sub
            
            
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

            
        elif  multi_classes_to_replace is not None and class_to_replace is not None and module==None:
            print(train_set_for_test[0])
            len_train = len(train_set_for_test)
            all_data_indices = range(len_train)
            forget_dict = {}
            forget_indices_for_test_all = []
            for cls in multi_classes_to_replace:
                forget_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) == cls)
                forget_indices_for_test_all.append(forget_indices_for_test)
                forget_set_for_test_sub = Subset(train_set_for_test, forget_indices_for_test) 
                forget_dict[str(cls)]=forget_set_for_test_sub
                
            # retain_indices_for_test_all = list(set(all_data_indices) - set(forget_indices_for_test_all))
            retain_indices_for_test_all = np.setdiff1d(all_data_indices, forget_indices_for_test_all)

            retain_set = Subset(train_set_for_test, retain_indices_for_test_all)
            
            #retain_indices_val = np.flatnonzero(np.array(val_set.targets) != class_to_replace)
            delete_val_indices_for_test_all = []
            for cls in multi_classes_to_replace:
                delete_val_indices = np.flatnonzero(np.array(val_set.targets) == cls)
                delete_val_indices_for_test_all.append(delete_val_indices)
                
            len_val = len(val_set)
            all_data_indices_val = range(len_val)
            val_indices_for_all = np.setdiff1d(all_data_indices_val, delete_val_indices_for_test_all)

            val_set_all = Subset(val_set, val_indices_for_all)
            
            print("forget_indices_for_test_all",forget_indices_for_test_all)
            print("retain_indices_for_test_all",retain_indices_for_test_all)
            print("retain_indices_for_test_all",val_indices_for_all)
            sets = {
                    "retain": retain_set,
                    "forget":forget_dict,   
                    "val":val_set_all
                    }
            return sets
        
        elif  multi_classes_to_replace is not None and class_to_replace is not None and module=="classwise":
            print(train_set_for_test[0],"####################classwise")
                #train_set,train_set_for_test,val_set
            
            len_train = len(train_set_for_test)
            all_data_indices = range(len_train)
            
            #forget_dict_test
            forget_dict = {}
            forget_indices_for_test_all = []
            
            for cls in multi_classes_to_replace:
                forget_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) == cls)
                forget_indices_for_test_all.extend(forget_indices_for_test)
                forget_set_for_test_sub = Subset(train_set_for_test, forget_indices_for_test) 
                forget_dict[str(cls)]=forget_set_for_test_sub
            
            #forget_set
            forget_set = Subset(train_set, forget_indices_for_test_all)
            #retain_set
            retain_indices_for_test_all = np.setdiff1d(all_data_indices, forget_indices_for_test_all)
            retain_set = Subset(train_set, retain_indices_for_test_all)
            #retain_set_for_test
            retain_set_for_test = Subset(train_set_for_test, retain_indices_for_test_all)
            
            delete_val_indices_for_test_all = []
            for cls in multi_classes_to_replace:
                delete_val_indices = np.flatnonzero(np.array(val_set.targets) == cls)
                delete_val_indices_for_test_all.append(delete_val_indices)
                
            len_val = len(val_set)
            all_data_indices_val = range(len_val)
            val_indices_for_all = np.setdiff1d(all_data_indices_val, delete_val_indices_for_test_all)

            #val_set
            val_set_all = Subset(val_set, val_indices_for_all)

            sets = {
                    "retain": retain_set,
                    "retain_for_test":retain_set_for_test,
                    "forget":forget_set,
                    "forget_dict_test":forget_dict,
                    "val":val_set_all,
                    }
            return sets
        

            

        elif class_to_replace is not None and num_indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        elif class_to_replace is None and num_indexes_to_replace is None:
            print("original")

            print(f"Train set size for original model: {len(train_set)}")
            print(f"Test set size for original model: {len(val_set)}")
            sets = {
                    "train": train_set,
                    "train_for_test": train_set_for_test,
                    "val":val_set,
                    }
            return sets
    
        elif class_to_replace is not None and num_indexes_to_replace is None:
            
            
            #class-wise
            forget_indices = np.flatnonzero(np.array(train_set.targets) == class_to_replace)
            print("forget_indices",forget_indices)
            retain_indices = np.flatnonzero(np.array(train_set.targets) != class_to_replace)
            
            forget_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) == class_to_replace)
            print("forget_indices_for_test",forget_indices_for_test)
            retain_indices_for_test = np.flatnonzero(np.array(train_set_for_test.targets) != class_to_replace)
            
            
            forget_indices_val = np.flatnonzero(np.array(val_set.targets) == class_to_replace)
            print("forget_indices_val",forget_indices_val)
            retain_indices_val = np.flatnonzero(np.array(val_set.targets) != class_to_replace)
            
            
            retain_set = Subset(train_set, retain_indices)
            forget_set = Subset(train_set, forget_indices)   
             
            forget_set_for_test = Subset(train_set_for_test, forget_indices_for_test) 
            retain_set_for_test = Subset(train_set_for_test, retain_indices_for_test)
            
            retain_set_val = Subset(val_set, retain_indices_val)
            forget_set_val = Subset(val_set, forget_indices_val)
            
            # perturbed_forget_set = Subset(new_train_set, forget_indices)
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
            print("data-wise")
            len_train = len(train_set)
            assert len_train == 50000, f"The number for data-wise of cifar10 setting is wrong."
            
            indexes = None
            forget_indices = None
            retain_indices = None
            
            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            print("forget rate: #####################", percent/100)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)
            forget_indices = indexes
            print("forget_indices",forget_indices)
            
            all_index = np.arange(len_train)
            # print("all_index",all_index[0])
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
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

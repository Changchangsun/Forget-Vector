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
    data_path: str=None,
    percent: int=10,
    arch: str="resnet18",
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
        else:
            print("Wrong")
        if multi_classes_to_replace is not None and num_indexes_to_replace is not None:
            print("data-wise")
            print(len(val_set))
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
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)
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
            print("data-wise")
            len_train = len(train_set)
            assert len_train == 13000, f"The number for data-wise of imagenet10 setting is wrong."

            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            print("forget rate: #####################", percent/100)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)
            forget_indices = indexes
            print("forget_indices",forget_indices)
            print(int(len_train*0.1),"forget length")
        
            all_index = np.arange(len_train)
            mask = ~np.isin(all_index, forget_indices)
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
            
            # ##随机取
            indexes = None
            forget_indices = None
            retain_indices = None
            
            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            print("forget rate: #####################", percent/100)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)#replace=False：表示抽样时不允许重复
            # indexes = rng.choice(indexes, size=int(len_train*0.1), replace=False)#replace=False：表示抽样时不允许重复
            forget_indices = indexes
            print("forget_indices",forget_indices)
            
            # 使用 np.isin 来找到要保留的元素
            all_index = np.arange(len_train)
            # print("all_index",all_index[0])
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
            # 应用掩码来创建新的数组
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
            
            #retain_indices_val = np.flatnonzero(np.array(val_set.targets) != class_to_replace)
            delete_val_indices_for_test_all = []
            for cls in multi_classes_to_replace:
                delete_val_indices = np.flatnonzero(np.array(val_set.targets) == cls)
                delete_val_indices_for_test_all.append(delete_val_indices)
                
            len_val = len(val_set)
            all_data_indices_val = range(len_val)
            val_indices_for_all = np.setdiff1d(all_data_indices_val, delete_val_indices_for_test_all)

            #val_set
            val_set_all = Subset(val_set, val_indices_for_all)
            
            # print("forget_indices_for_test_all",forget_indices_for_test_all)
            # print("retain_indices_for_test_all",retain_indices_for_test_all)
            # print("retain_indices_for_test_all",val_indices_for_all)
            sets = {
                    "retain": retain_set,
                    "retain_for_test":retain_set_for_test,
                    "forget":forget_set,
                    "forget_dict_test":forget_dict,
                    "val":val_set_all,
                    }
            return sets
        
        # sets = {
        #             "retain": retain_set,
        #             "forget":forget_set,
        #             "retain_for_test":retain_set_for_test,
        #             "forget_for_test":forget_set_for_test,
        #             "val": val_set,
        #             "val_retain": retain_set_val,
        #             "val_forget": forget_set_val,
        #             "retain_adv": retain_set_adv,
        #             "forget_adv": forget_set_adv,
        #             "val_adv": val_set_adv,    
        #             "val_retain_adv": retain_set_val_adv,
        #             "val_forget_adv":forget_set_val_adv,
        #             }
        #     return sets
            
                
            


                
            
        
        elif class_to_replace is not None and num_indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        elif class_to_replace is None and num_indexes_to_replace is None:
            print("original")
            # if single>=0:
            #     print("Evaluate the single class acc. ####")
            #     val_indices = np.flatnonzero(np.array(train_set.targets) == single)
            #     single_train_set = Subset(train_set, val_indices)
            #     print("len of single_train_set",len(single_train_set))
                
            #     loaders = {
            #         "train": train_loader,
            #         "val": DataLoader(
            #         single_train_set, batch_size=batch_size, num_workers=4, shuffle=False
            #     ),
            #     }
            #     return loaders
            print(f"Train set size for original model: {len(train_set)}")
            print(f"Test set size for original model: {len(val_set)}")
            sets = {
                    "train": train_set,
                    "train_for_test": train_set_for_test,
                    "val":val_set,
                    }
            return sets
    
        elif class_to_replace is not None and num_indexes_to_replace is None:
            print("class-wise")
            # new_train_set = copy.deepcopy(train_set)
            # # Assuming new_train_set.targets is a list
            # new_train_set.targets = np.array(new_train_set.targets)
            # # Perform the operation
            # new_train_set.targets = (new_train_set.targets + 1) % 10
            # # Convert back to a list if needed
            # new_train_set.targets = new_train_set.targets.tolist()
            
            
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
            
            # ##随机取
            indexes = None
            forget_indices = None
            retain_indices = None
            
            indexes = range(len_train)
            rng = np.random.RandomState(seed)
            print("forget rate: #####################", percent/100)
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)#replace=False：表示抽样时不允许重复
            # indexes = rng.choice(indexes, size=int(len_train*0.1), replace=False)#replace=False：表示抽样时不允许重复
            forget_indices = indexes
            print("forget_indices",forget_indices)
            
            # 使用 np.isin 来找到要保留的元素
            all_index = np.arange(len_train)
            # print("all_index",all_index[0])
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
            # 应用掩码来创建新的数组
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

    elif dataset == "cifar10_scc_transfer2":
        
        if arch=="vit":
            train_transform = transforms.Compose(
                [
                transforms.Resize((224, 224)),            # 调整图像大小为 224x224
                transforms.RandomCrop(224, padding=4),    # 随机裁剪并填充
                transforms.RandomHorizontalFlip(),        # 随机水平翻转
                transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),            # 调整图像大小为 224x224
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
        # 加载本地训练集
        # # train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        # # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        # train_set_path = ImageFolderWithPaths(os.path.join(data_path, 'train'), transform=train_transform)
        # train_loader_path = torch.utils.data.DataLoader(train_set_path, batch_size=batch_size, shuffle=True, num_workers=4)
        # all_path=[]
        # all_target=[]
        
        #train set 1
        train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        
        train_set_for_test = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=test_transform)

        #val set 2
        val_set = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=test_transform)
        
        
        train_set_adv = None
        val_set_adv = None
        
        if adv is not None:
            print("adv",adv,"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            #"/egr/research-optml/sunchan5/MU/Unlearn-Sparse/datasets/ImageNet10_pgd_attack_0.3_0.1_40/train_same_folder"
            adv_path = adv
            #train set adv 3
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train_same_folder"), transform=test_transform)
            # train_loader_adv = torch.utils.data.DataLoader(train_set_adv, batch_size=batch_size, shuffle=False, num_workers=4)
            #val set adv 4
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"test_same_folder"), transform=test_transform)
            # val_loader_adv = torch.utils.data.DataLoader(val_set_adv, batch_size=batch_size, shuffle=False, num_workers=4)
        elif cor is not None:
            print("cor",cor,"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            #"/egr/research-optml/sunchan5/MU/Unlearn-Sparse/datasets/ImageNet10_pgd_attack_0.3_0.1_40/train_same_folder"
            adv_path = cor
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train",cor_type,level), transform=test_transform)
            # train_loader_adv = torch.utils.data.DataLoader(train_set_adv, batch_size=batch_size, shuffle=False, num_workers=4)
            #validation_same_folder
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"test",cor_type,level), transform=test_transform)
            #
        else:
            print("Wrong")
        
        if multi_classes_to_replace is not None:
                        
            # Classes to forget
            classes_to_forget = multi_classes_to_replace
            all_classes = np.arange(num_classes) 
            classes_to_retain = np.setdiff1d(all_classes, classes_to_forget)

            print("classes_to_forget:",classes_to_forget,"#####################")
            print("classes_to_retain:", classes_to_retain,"#####################")
            
            # Find indices where the target is in classes_to_forget
            forget_indices = np.flatnonzero(np.isin(train_set.targets, classes_to_forget))
            # Find indices where the target is in classes_to_retain
            retain_indices = np.flatnonzero(np.isin(train_set.targets, classes_to_retain))
            # Optionally, print or use these indices
            print("Forget indices:", forget_indices)
            print("Retain indices:", retain_indices)
            
            
            # Find indices where the target is in classes_to_forget
            forget_indices_for_test = np.flatnonzero(np.isin(train_set.targets, classes_to_forget))
            # Find indices where the target is in classes_to_retain
            retain_indices_for_test = np.flatnonzero(np.isin(train_set.targets, classes_to_retain))
            # Optionally, print or use these indices
            print("forget_indices_for_test:", forget_indices_for_test)
            print("retain_indices_for_test:", retain_indices_for_test)
            
            
            # Find indices where the target is in classes_to_forget
            forget_indices_val = np.flatnonzero(np.isin(val_set.targets, classes_to_forget))
            # Find indices where the target is in classes_to_retain
            retain_indices_val = np.flatnonzero(np.isin(val_set.targets, classes_to_retain))
            # Optionally, print or use these indices
            print("forget_indices_val:", forget_indices_val)
            print("retain_indices_val:", retain_indices_val)
            
            forget_set = Subset(train_set, forget_indices)
            retain_set = Subset(train_set, retain_indices)
            forget_set_for_test = Subset(train_set_for_test, forget_indices_for_test) 
            retain_set_for_test = Subset(train_set_for_test, retain_indices_for_test)
            
            forget_set_val = Subset(val_set, forget_indices_val)
            retain_set_val = Subset(val_set, retain_indices_val)
            
            print(f"retain set size: {len(retain_set)}")
            print(f"forget set size: {len(forget_set)}")
            print(f"retain_for_test set size: {len(retain_set_for_test)}")
            print(f"forget_for_test set size: {len(forget_set_for_test)}")
            print(f"val set size: {len(val_set)}")
            print(f"val_retain set size: {len(retain_set_val)}")
            print(f"val_forget set size: {len(forget_set_val)}")
            
            sets = {
                    "retain": retain_set,
                    "forget":forget_set,
                    "retain_for_test":retain_set_for_test,
                    "forget_for_test":forget_set_for_test,
                    "val": val_set,
                    "val_retain": retain_set_val,
                    "val_forget": forget_set_val,
                    }
            return sets
    
    
    elif dataset == "imagenet10_transfer2":
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
            print("adv",adv,"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            #"/egr/research-optml/sunchan5/MU/Unlearn-Sparse/datasets/ImageNet10_pgd_attack_0.3_0.1_40/train_same_folder"
            adv_path = adv
            #train set adv 3
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train_same_folder"), transform=validation_transform)
            # train_loader_adv = torch.utils.data.DataLoader(train_set_adv, batch_size=batch_size, shuffle=False, num_workers=4)
            #val set adv 4
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"validation_same_folder"), transform=validation_transform)
            # val_loader_adv = torch.utils.data.DataLoader(val_set_adv, batch_size=batch_size, shuffle=False, num_workers=4)
        elif cor is not None:
            print("cor",cor,"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            #"/egr/research-optml/sunchan5/MU/Unlearn-Sparse/datasets/ImageNet10_pgd_attack_0.3_0.1_40/train_same_folder"
            adv_path = cor
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path,"train",cor_type,level), transform=validation_transform)
            # train_loader_adv = torch.utils.data.DataLoader(train_set_adv, batch_size=batch_size, shuffle=False, num_workers=4)
            #validation_same_folder
            val_set_adv = datasets.ImageFolder(os.path.join(adv_path,"validation",cor_type,level), transform=validation_transform)
            #
        else:
            print("Wrong")
        
        if multi_classes_to_replace is not None:
                        
            # Classes to forget
            classes_to_forget = multi_classes_to_replace
            all_classes = np.arange(num_classes) 
            classes_to_retain = np.setdiff1d(all_classes, classes_to_forget)

            print("classes_to_forget:",classes_to_forget,"#####################")
            print("classes_to_retain:", classes_to_retain,"#####################")
            
            # Find indices where the target is in classes_to_forget
            forget_indices = np.flatnonzero(np.isin(train_set.targets, classes_to_forget))
            # Find indices where the target is in classes_to_retain
            retain_indices = np.flatnonzero(np.isin(train_set.targets, classes_to_retain))
            # Optionally, print or use these indices
            print("Forget indices:", forget_indices)
            print("Retain indices:", retain_indices)
            
            
            # Find indices where the target is in classes_to_forget
            forget_indices_for_test = np.flatnonzero(np.isin(train_set.targets, classes_to_forget))
            # Find indices where the target is in classes_to_retain
            retain_indices_for_test = np.flatnonzero(np.isin(train_set.targets, classes_to_retain))
            # Optionally, print or use these indices
            print("forget_indices_for_test:", forget_indices_for_test)
            print("retain_indices_for_test:", retain_indices_for_test)
            
            
            # Find indices where the target is in classes_to_forget
            forget_indices_val = np.flatnonzero(np.isin(val_set.targets, classes_to_forget))
            # Find indices where the target is in classes_to_retain
            retain_indices_val = np.flatnonzero(np.isin(val_set.targets, classes_to_retain))
            # Optionally, print or use these indices
            print("forget_indices_val:", forget_indices_val)
            print("retain_indices_val:", retain_indices_val)
            
            forget_set = Subset(train_set, forget_indices)
            retain_set = Subset(train_set, retain_indices)
            forget_set_for_test = Subset(train_set_for_test, forget_indices_for_test) 
            retain_set_for_test = Subset(train_set_for_test, retain_indices_for_test)
            
            forget_set_val = Subset(val_set, forget_indices_val)
            retain_set_val = Subset(val_set, retain_indices_val)
            
            print(f"retain set size: {len(retain_set)}")
            print(f"forget set size: {len(forget_set)}")
            print(f"retain_for_test set size: {len(retain_set_for_test)}")
            print(f"forget_for_test set size: {len(forget_set_for_test)}")
            print(f"val set size: {len(val_set)}")
            print(f"val_retain set size: {len(retain_set_val)}")
            print(f"val_forget set size: {len(forget_set_val)}")
            
            sets = {
                    "retain": retain_set,
                    "forget":forget_set,
                    "retain_for_test":retain_set_for_test,
                    "forget_for_test":forget_set_for_test,
                    "val": val_set,
                    "val_retain": retain_set_val,
                    "val_forget": forget_set_val,
                    }
            return sets
            
                    
        if class_to_replace is not None and num_indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        elif class_to_replace is None and num_indexes_to_replace is None:
            print("original")   
            # if single>=0:
            #     print("Evaluate the single class acc. ####")
            #     val_indices = np.flatnonzero(np.array(train_set.targets) == single)
            #     single_train_set = Subset(train_set, val_indices)
            #     print("len of single_train_set",len(single_train_set))
                
            #     loaders = {
            #         "train": train_loader,
            #         "val": DataLoader(
            #         single_train_set, batch_size=batch_size, num_workers=4, shuffle=False
            #     ),
            #     }
            #     return loaders
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
            indexes = rng.choice(indexes, size=int(len_train*(percent/100)), replace=False)#replace=False：表示抽样时不允许重复
            forget_indices = indexes
            print("forget_indices",forget_indices)
            print(int(len_train*0.1),"forget length")
        
            # 使用 np.isin 来找到要保留的元素
            all_index = np.arange(len_train)
            # print("all_index",all_index[0])
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
            # 应用掩码来创建新的数组
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
    
    elif dataset == "imagenet10_pgd":
    #     train_transform = transforms.Compose([
    #                 torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
    #                 torchvision.transforms.RandomResizedCrop((224, 224)),
    #                 torchvision.transforms.RandomHorizontalFlip(),
    #                 torchvision.transforms.ToTensor(),
    #             ]
    # )
        train_transform = transforms.Compose([
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((224, 224)),
                    # torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                ]
    )
        validation_transform = transforms.Compose([
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((224, 224)),
                    # torchvision.transforms.CenterCrop((224, 224)),
                    torchvision.transforms.ToTensor(),
                ])
        """"
    transform = transforms.Compose([
    transforms.Resize((224,224)),  #
    transforms.ToTensor(),  # 将图像转换为PyTorch张量 这个操作还会对图像进行归一化，将像素值从0-255的范围映射到0-1之间
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
])
        """
                
        # 加载本地训练集
        # train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        train_set_path = ImageFolderWithPaths(os.path.join(data_path, 'train'), transform=train_transform)
        train_loader_path = torch.utils.data.DataLoader(train_set_path, batch_size=batch_size, shuffle=False, num_workers=4)
        
        val_set_path = ImageFolderWithPaths(os.path.join(data_path, 'validation'), transform=validation_transform)
        val_loader_path = torch.utils.data.DataLoader(val_set_path, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_path=[]
        all_target=[]
        
        train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
        
        val_set = datasets.ImageFolder(os.path.join(data_path, 'validation'), transform=validation_transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        # 加载本地测试集
        # test_set = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=validation_transform)
        # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

        # for images, targets, paths in train_loader_path:
        #     # print("1")
        #     print(images[0])#（0,1）
        #     break

        # 打印数据集大小
        print(f"Training set size: {len(train_set)}")
        print(f"Validation set size: {len(val_set)}")
        loaders = {
            "train": train_loader_path,
            "val":val_loader_path,
            # "test":test_loader,
        }
        return loaders
    elif dataset == "cifar10_scc_pgd":
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
                
        # 加载本地训练集
        # train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        train_set_path = ImageFolderWithPaths(os.path.join(data_path, 'train'), transform=train_transform)
        train_loader_path = torch.utils.data.DataLoader(train_set_path, batch_size=batch_size, shuffle=True, num_workers=4)
        all_path=[]
        all_target=[]
        val_set_path =  ImageFolderWithPaths(os.path.join(data_path, 'test'), transform=test_transform)
        val_loader_path = torch.utils.data.DataLoader(val_set_path, batch_size=batch_size, shuffle=False, num_workers=4)
        
        train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        if phase=="train":
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            print("shuffle is false")
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
            
        val_set = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=test_transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        # 加载本地测试集
        # test_set = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=validation_transform)
        # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


        # 打印数据集大小
        print(f"Training set size: {len(train_set)}")
        print(f"Validation set size: {len(val_set)}")
        # print(f"test set size: {len(test_set)}")
        
        num_training_set = len(train_set)#num_indexes_to_replace
        loaders = {
            "train": train_loader_path,
            "val":val_loader_path,
            # "test":test_loader,
        }
        return loaders

    elif dataset == "Imagenet":
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
                
        # 加载本地训练集
        # train_set = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        train_set_path = ImageFolderWithPaths(os.path.join(data_path, 'train_class'), transform=train_transform)
        train_loader_path = torch.utils.data.DataLoader(train_set_path, batch_size=batch_size, shuffle=False, num_workers=4)
        all_path=[]
        all_target=[]
        
        train_set = datasets.ImageFolder(os.path.join(data_path, 'train_class'), transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        
        train_set_adv = None
        train_loader_adv = None
        if adv==1:
            adv_path = "/egr/research-optml/sunchan5/MU/Unlearn-Sparse/datasets/ImageNet10_Corruptions/ImageNet10_distorted/train/motion_blur/5"
            train_set_adv = datasets.ImageFolder(os.path.join(adv_path), transform=validation_transform)
            train_loader_adv = torch.utils.data.DataLoader(train_set_adv, batch_size=batch_size, shuffle=True, num_workers=4)
        # else:
        #     val_set = datasets.ImageFolder(os.path.join(data_path, 'validation'), transform=validation_transform)
        #     val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # val_set = None
        # val_loader = None
        val_set = datasets.ImageFolder(os.path.join(data_path, 'val_class'), transform=validation_transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        # 加载本地测试集
        # test_set = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=validation_transform)
        # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


        # 打印数据集大小
        print(f"Training set size: {len(train_set)}")
        print(f"Validation set size: {len(val_set)}")
        # print(f"test set size: {len(test_set)}")
        
        num_training_set = len(train_set)#num_indexes_to_replace
    elif dataset == "tiny_imagenet":
        train_set = load_dataset(
            "Maysee/tiny-imagenet", use_auth_token=True, split="train", cache_dir=path
        )
        validation_set = load_dataset(
            "Maysee/tiny-imagenet", use_auth_token=True, split="valid", cache_dir=path
        )

        def train_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.RandomCrop(64, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

    elif dataset == "flowers102":
        train_set = load_dataset(
            "nelorth/oxford-flowers", use_auth_token=True, split="train", cache_dir=path
        )
        validation_set = load_dataset(
            "nelorth/oxford-flowers", use_auth_token=True, split="test", cache_dir=path
        )

        def train_transform(examples):
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.RandomCrop((224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
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
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [transform(x) for x in examples["image"]]
            return examples

    else:
        raise NotImplementedError
    # train_set.set_transform(transform=train_transform)
    # validation_set.set_transform(transform=validation_transform)

    if class_to_replace is not None and num_indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    elif class_to_replace is None and num_indexes_to_replace is None:
        #original loader
        if single>=0:
            print("Evaluate the single class acc. ####")
            val_indices = np.flatnonzero(np.array(train_set.targets) == single)
            single_train_set = Subset(train_set, val_indices)
            print("len of single_train_set",len(single_train_set))
            
            loaders = {
                "train": train_loader,
                "val": DataLoader(
                single_train_set, batch_size=batch_size, num_workers=4, shuffle=False
            ),
            }
            return loaders
        loaders = {
            "train": train_loader,
            "val":val_loader,
            # "test":test_loader,
        }
        return loaders
    
    elif class_to_replace is not None and num_indexes_to_replace is None:
        #class-wise
        forget_indices = np.flatnonzero(np.array(train_set.targets) == class_to_replace)
        print("forget_indices",forget_indices)
        retain_indices = np.flatnonzero(np.array(train_set.targets) != class_to_replace)
        retain_set = Subset(train_set, retain_indices)
        forget_set = None
        if adv is not None:
            forget_set = Subset(train_set_adv, forget_indices)
        else:
            forget_set = Subset(train_set, forget_indices)
        
        # retain_indices_test = np.flatnonzero(np.array(test_set.targets) != class_to_replace)
        # test_set = Subset(test_set, retain_indices_test)
        
        forget_indices_val = np.flatnonzero(np.array(val_set.targets) == class_to_replace)
        print("forget_indices_val",forget_indices_val)
        retain_indices_val = np.flatnonzero(np.array(val_set.targets) != class_to_replace)
        retain_set_val = Subset(val_set, retain_indices_val)
        forget_set_val = Subset(val_set, forget_indices_val)
        
        print(f"After forget retain_set size: {len(retain_set)}")
        print(f"forget_set set size: {len(forget_set)}")
        print(f"val set size: {len(val_set)}")
        print(f"val_retain set size: {len(retain_set_val)}")
        print(f"val_forget set size: {len(forget_set_val)}")
        # print(f"test set size: {len(test_set)}")
        if phase=="train":
            loaders = {
                "retain": DataLoader(
                    retain_set, batch_size=batch_size, num_workers=4, shuffle=True
                ),#retain+forget
                "forget": DataLoader(
                    forget_set, batch_size=batch_size, num_workers=4, shuffle=False
                ),
                "val": DataLoader(
                    val_set, batch_size=batch_size, num_workers=4, shuffle=False
                ),
                "val_retain": DataLoader(
                    retain_set_val, batch_size=batch_size, num_workers=4, shuffle=False
                ),
                "val_forget": DataLoader(
                    forget_set_val, batch_size=batch_size, num_workers=4, shuffle=False
                )}
            # return loaders
        else:
            print("shuffle is false")
            loaders = {
            "retain": DataLoader(
                retain_set, batch_size=batch_size, num_workers=4, shuffle=False
            ),#retain+forget
            "forget": DataLoader(
                forget_set, batch_size=batch_size, num_workers=4, shuffle=False
            ),
            "val": DataLoader(
                val_set, batch_size=batch_size, num_workers=4, shuffle=False
            ),
            "val_retain": DataLoader(
                retain_set_val, batch_size=batch_size, num_workers=4, shuffle=False
            ),
            "val_forget": DataLoader(
                forget_set_val, batch_size=batch_size, num_workers=4, shuffle=False
            ),
            # "test": DataLoader(
            #     test_set, batch_size=batch_size, num_workers=4, shuffle=False
            # ),
        }
        return loaders
    elif num_indexes_to_replace is not None:
        print("data wise$$")
        #        num_training_set = len(train_set)#num_indexes_to_replace
        assert num_indexes_to_replace <= len(
            train_set
        ), f"Want to replace {num_indexes_to_replace} indexes but only {len(train_set)} samples in dataset"
        # ##均匀取
        # indexes = []
        # for i in range(10):
        #     index = range(1300*i,1300*(i+1))
        #     rng = np.random.RandomState(seed)
        #     index = rng.choice(index, size=130, replace=False)
        #     # indexes = indexes + index
        #     indexes = np.concatenate((indexes, index))
        #     # print(indexes,len(indexes))
        #     print(len(indexes))
        #     list=np.zeros(10)
        #     for i in indexes:
        #         # print(i)
        #         if all_target[int(i)]<1300:
        #             list[0]+=1
        #         elif all_target[int(i)]>=1300 and int(i)<2600:
        #             list[1]+=1
        #         elif all_target[int(i)]>=2600 and int(i)<3900:
        #             list[2]+=1
        #         elif all_target[int(i)]>=3900 and int(i)<5200:
        #             list[3]+=1
        #         elif all_target[int(i)]>=5200 and int(i)<6500:
        #             list[4]+=1
        #         elif all_target[int(i)]>=6500 and int(i)<7800:
        #             list[5]+=1
        #         elif all_target[int(i)]>=7800 and int(i)<9100:
        #             list[6]+=1
        #         elif all_target[int(i)]>=9100 and int(i)<10400:
        #             list[7]+=1
        #         elif all_target[int(i)]>=10400 and int(i)<11700:
        #             list[8]+=1
        #         elif all_target[int(i)]>=11700 and int(i)<13000:
        #             list[9]+=1
        #     print(list)
            
        # ##随机取
        indexes = None
        forget_indices = None
        retain_indices = None
        
        if dataset == "imagenet10":
            indexes = range(13000)
            rng = np.random.RandomState(seed)
            indexes = rng.choice(indexes, size=1300, replace=False)
            print(indexes)
            forget_indices = indexes
            print("forget_indices",forget_indices)
        
            # 使用 np.isin 来找到要保留的元素
            all_index = np.arange(13000)
            print("all_index",all_index[0])
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
            # 应用掩码来创建新的数组
            retain_indices = all_index[mask]
            
        elif dataset == "cifar10_scc":
            indexes = range(50000)
            rng = np.random.RandomState(seed)
            indexes = rng.choice(indexes, size=5000, replace=False)
            print(indexes)
            forget_indices = indexes
            print("forget_indices",forget_indices)
        
            # 使用 np.isin 来找到要保留的元素
            all_index = np.arange(50000)
            print("all_index",all_index[0])
            mask = ~np.isin(all_index, forget_indices)
            print("mask",mask)
            # 应用掩码来创建新的数组
            retain_indices = all_index[mask]
        
        # list=np.zeros(10)
        # replace_index_images = open("/egr/research-optml/sunchan5/MU/Unlearn-Sparse/trained_models/ImageNet10/RT/resnet10/datawise_wise/v_random/replace_images_path.txt","w")
        # retain_index_images = open("/egr/research-optml/sunchan5/MU/Unlearn-Sparse/trained_models/ImageNet10/RT/resnet10/datawise_wise/v_random/retain_images_path.txt","w")

        # for i in indexes:
        #     # print(i)
        #     replace_index_images.write(str(all_path[int(i)])+"\t"+str(all_target[int(i)])+"\n")
        #     if all_target[int(i)]==0:
        #         list[0]+=1
        #     elif all_target[int(i)]==1:
        #         list[1]+=1
        #     elif all_target[int(i)]==2:
        #         list[2]+=1
        #     elif all_target[int(i)]==3:
        #         list[3]+=1
        #     elif all_target[int(i)]==4:
        #         list[4]+=1
        #     elif all_target[int(i)]==5:
        #         list[5]+=1
        #     elif all_target[int(i)]==6:
        #         list[6]+=1
        #     elif all_target[int(i)]==7:
        #         list[7]+=1
        #     elif all_target[int(i)]==8:
        #         list[8]+=1
        #     elif all_target[int(i)]==9:
        #         list[9]+=1
        # print("list",list)
        # replace_index_images.close()

        # forget_indices = indexes
        # print("forget_indices",forget_indices)
    
        # # 使用 np.isin 来找到要保留的元素
        # all_index = np.arange(13000)
        # print("all_index",all_index[0])
        # mask = ~np.isin(all_index, forget_indices)
        # print("mask",mask)
        # # 应用掩码来创建新的数组
        # retain_indices = all_index[mask]
        
        # for i in retain_indices:
        #     # print(i)
        #     retain_index_images.write(str(all_path[int(i)])+"\t"+str(all_target[int(i)])+"\n")
        # retain_index_images.close()
        # retain_indices = 
        
        # retain_indices = np.flatnonzero(np.array(train_set.targets) != class_to_replace)
        
        # forget_indices = torch.ones_like(train_subset_indices) - train_subset_indices
        # train_subset_indices = torch.nonzero(train_subset_indices)

        # forget_indices = torch.nonzero(forget_indices)
        retain_set = Subset(train_set, retain_indices)
        
        forget_set = None
        if adv is not None:
            forget_set = Subset(train_set_adv, forget_indices)
        else:
            forget_set = Subset(train_set, forget_indices)
            
        # forget_set = Subset(train_set, forget_indices)
        
        # retain_indices_test = np.flatnonzero(np.array(test_set.targets) != class_to_replace)
        # test_set = Subset(test_set, retain_indices_test)
        
        print(f"After forget retain_set size: {len(retain_set)}")
        print(f"forget_set set size: {len(forget_set)}")
        print(f"val set size: {len(val_set)}")
        # print(f"test set size: {len(test_set)}")
        if phase == "train":
            loaders = {
                "retain": DataLoader(
                    retain_set, batch_size=batch_size, num_workers=4, shuffle=True
                ),#retain+forget
                "forget": DataLoader(
                    forget_set, batch_size=batch_size, num_workers=4, shuffle=False  #####
                ),
                "val": DataLoader(
                    val_set, batch_size=batch_size, num_workers=4, shuffle=False
                ),
                # "test": DataLoader(
                #     test_set, batch_size=batch_size, num_workers=4, shuffle=False
                # ),
            }
        else:
            loaders = {
                "retain": DataLoader(
                    retain_set, batch_size=batch_size, num_workers=4, shuffle=False
                ),#retain+forget
                "forget": DataLoader(
                    forget_set, batch_size=batch_size, num_workers=4, shuffle=False  #####
                ),
                "val": DataLoader(
                    val_set, batch_size=batch_size, num_workers=4, shuffle=False
                ),
                # "test": DataLoader(
                #     test_set, batch_size=batch_size, num_workers=4, shuffle=False
                # ),
            }
        return loaders
        
    
    


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

from .fisher import fisher, fisher_new
from .FT import FT, FT_l1
from .FT_prune import FT_prune
from .FT_prune_bi import FT_prune_bi
from .GA import GA, GA_l1
from .NegGrad_plus import NegGrad_plus
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint,load_unlearn_checkpoint_original,load_unlearn_checkpoint_saliency,load_original_checkpoint,save_unlearn_checkpoint_baseline
from .retrain import retrain
from .retrain_ls import retrain_ls
from .retrain_sam import retrain_sam
from .Wfisher import Wfisher
from .RL import RL
from .IU import IU
from .SCRUB import SCRUB

def raw(data_loaders, model, criterion, args):
    print("raw@")
    pass

# def retrain(data_loaders, model, criterion, optimizer, epoch, args):
#     retain_loader = data_loaders["retain"]
#     print("retain_loader###")
#     return train(retain_loader, model, criterion, optimizer, epoch, args)

def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "RL":
        print("Using RL method#####################")
        return RL
    elif name == "GA":
        return GA
    #scc
    elif name=="SCRUB":
        return SCRUB
    elif name == "NegGrad_plus":
        return NegGrad_plus
    elif name == "FT":
        return FT
    elif name == "IU":
        return IU
    elif name == "FT_l1":
        return FT_l1
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        print("1 retrain")
        print("2222")
        return retrain
    elif name == "fisher_new":
        return fisher_new
    elif name == "wfisher":
        return Wfisher
    elif name == "FT_prune":
        return FT_prune
    elif name == "FT_prune_bi":
        return FT_prune_bi
    elif name == "GA_l1":
        return GA_l1
    elif name == "retrain_ls":
        return retrain_ls
    elif name == "retrain_sam":
        return retrain_sam
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")

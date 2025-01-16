from trainer import train

from .impl import iterative_unlearn
import unlearn

@iterative_unlearn
def retrain(data_loaders, model, criterion, optimizer, epoch, args):
    print("2 retrain",epoch)
    retain_loader = data_loaders["retain"]
    print("3 retain_loader###",epoch)
    return train(retain_loader, model, criterion, optimizer, epoch, args)

# train_acc = unlearn_iter_func(
#                 data_loaders, model, criterion, optimizer, epoch, args
#             )

import torch

import delta_utils
from imagenet import get_x_y_from_data_dict
from tqdm import tqdm

def validate(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    total_accuracy = 0.0
    total_samples = 0

    model.eval()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(val_loader):
            image, target = get_x_y_from_data_dict(data, device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = delta_utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        # count=0
            #         for i, (image, target, source) in enumerate(tqdm(combined_dataset_loader, desc='main_delta_2loss.py')):            
        for i, (image, target) in enumerate(tqdm(val_loader)):
            image = image.cuda()
            target = target.cuda()
            
            # count+=image.size(0)

            # compute output
            with torch.no_grad():
                output = model(image)
                # loss = criterion(output, target)
            
            prec1 = delta_utils.accuracy(output, target)[0]  

            total_accuracy += prec1.item() * image.size(0)  
            total_samples += image.size(0)
    
    average_accuracy = total_accuracy / total_samples
    return average_accuracy


def delta_validate(val_loader, model, criterion, delta, args):
    """
    Run evaluation
    """
    losses = delta_utils.AverageMeter()
    top1 = delta_utils.AverageMeter()

    # switch to evaluate mode
    model.eval()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(val_loader):
            image, target = get_x_y_from_data_dict(data, device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = delta_utils.accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        count=0
        # for i, (image, target) in enumerate(val_loader):
        for i, (image, target) in enumerate(tqdm(val_loader)):

            delta = delta.cuda()
            image = image.cuda()
            image = image+delta
            target = target.cuda()
            count+=image.size(0)

            # compute output
            with torch.no_grad():
                output = model(image)
                # loss = criterion(output, target)

            output = output.float()

            # measure accuracy and record loss
            prec1 = delta_utils.accuracy(output.data, target)[0]
            top1.update(prec1.item(), image.size(0))

    return top1.avg

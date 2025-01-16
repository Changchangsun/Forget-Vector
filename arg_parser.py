import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")

    ##################################### Dataset #################################################
    parser.add_argument("--data", type=str, default="../data", help="location of the data corpus")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument("--input_size", type=int, default=32, help="size of input images")
    parser.add_argument("--data_dir", type=str, default="./tiny-imagenet-200", help="dir to tiny-imagenet")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    
    ##################################### Architecture ############################################
    parser.add_argument("--arch", type=str, default="resnet18", help="model architecture")
    parser.add_argument("--imagenet_arch", action="store_true", help="architecture for imagenet size samples")
    parser.add_argument("--train_y_file", type=str, default="./labels/train_ys.pth", help="labels for training files")
    parser.add_argument("--val_y_file", type=str, default="./labels/val_ys.pth", help="labels for validation files")
    
    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument("--train_seed", default=1, type=int, help="seed for training (default value same as args.seed)")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--workers", type=int, default=4, help="number of workers in dataloader")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument("--save_dir",help="The directory used to save the trained models",default=None,type=str)
    parser.add_argument("--mask", type=str, default=None, help="sparse model")
    parser.add_argument("--phase", type=str, default="train", help="train or test phase")
    parser.add_argument("--model_path", type=str, default=None, help="the path of original model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--epochs", default=182, type=int, help="number of total epochs to run")
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=20, type=int, help="print frequency")#50
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument("--no_aug", action="store_true", default=False, help="No augmentation in training dataset (transformation).")
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")
    
    ##################################### Unlearn setting #################################################
    parser.add_argument("--unlearn", type=str, default="retrain", help="method to unlearn")
    parser.add_argument("--unlearn_lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--unlearn_epochs", default=10, type=int, help="number of total epochs for unlearn to run")
    parser.add_argument("--num_indexes_to_replace", type=int, default=None, help="Number of data to forget")
    parser.add_argument("--class_to_replace", type=int, default=None,help="Specific class to forget")
    parser.add_argument("--multi_classes_to_replace", type=int, nargs='+', default = None, help="Specific class(es) to forget")
    parser.add_argument("--class_to_replace_random_label", type=int, help="Specific class to forget of the random label unlearn method")
    parser.add_argument("--schedule", type=int, default = 1, help="Specific class to forget of the random label unlearn method")
    parser.add_argument("--indexes_to_replace", type=list, default=None, help="Specific index data to forget")
    parser.add_argument("--all_delta", type=str, default=None, help="Specific index data to forget")

    parser.add_argument("--adv", type=str, default=None, help="1 = adv samples")
    parser.add_argument("--cor", type=str, default=None, help="1 = cor samples")
    parser.add_argument("--cor_type", type=str, default=None, help="1 = cor samples")
    parser.add_argument("--level", type=str, default=None, help="1 = cor samples")
    parser.add_argument("--single", type=int, default=-1, help="single class val?")
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")
    parser.add_argument("--mask_path", default=None, type=str, help="the path of saliency map")
    parser.add_argument("--result_path", type=str, default=None, help="record the result in txt file.")
    parser.add_argument("--test_method", type=str, default=None, help="record the result of cam.")
    parser.add_argument("--delta_path", type=str, default=None, help="record the result in txt file.")
    parser.add_argument("--exp_name", type=str, default=None, help="record the result in wandb.")
    parser.add_argument("--Tau", default=0.0, type=float, help="unlearn noise")
    parser.add_argument("--Lambda", default=1.0, type=float, help="unlearn noise")
    #Alpha
    parser.add_argument("--Alpha", default=1.0, type=float, help="unlearn noise")
    parser.add_argument("--Beta", default=1.0, type=float, help="unlearn noise")
    #epsilon
    parser.add_argument("--epsilon", default=0.01, type=float, help="unlearn noise")
    #init_delta
    parser.add_argument("--init_delta", default=0, type=str, help="unlearn noise")
    parser.add_argument("--percent", default=10, type=int, help="unlearn noise")
    parser.add_argument("--alpha_iu", default=1.0, type=float, help="unlearn noise")
    parser.add_argument("--beta_neggrad_plus", default=0.5, type=float, help="unlearn noise")
    parser.add_argument("--scrub_forget_epoch", default=2, type=float, help="unlearn noise")
    parser.add_argument("--case_w", type=int, default = 0, help="Specific class to forget of the random label unlearn method")
    parser.add_argument("--module", default=None, type=str, help="unlearn noise")
    parser.add_argument("--test_epoch_s", type=int, default=0, help="single class val?")
    parser.add_argument("--test_epoch_e", type=int, default=10, help="single class val?")
    parser.add_argument("--test_epoch_grad_cam_delta", type=int, default=10, help="single class val?")
    parser.add_argument("--test_epoch_grad_cam", type=int, default=10, help="single class val?")
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principal component of cam_weights*activations')

    ##################################### Attack setting #################################################
    parser.add_argument("--attack", type=str, default="backdoor", help="method to unlearn")
    parser.add_argument("--trigger_size", type=int, default=4, help="The size of trigger of backdoor attack")

    return parser.parse_args()

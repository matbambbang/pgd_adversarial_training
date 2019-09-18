import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from dataloader import get_cifar10_loaders
from utils import init_logger
from model.cifar10 import cifar_model
from container import trainer


parser = argparse.ArgumentParser("CIFAR10")
parser.add_argument("--model", type=str, default="conv", choices=["res", "wres", "conv"])
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--decay", type=float, default=0.0005)
parser.add_argument("--block", type=int, default=6)
parser.add_argument("--hist", type=eval, default=False)
parser.add_argument("--save", type=str, default="exp")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument("--norm", type=str, default="b")
parser.add_argument("--nesterov", type=eval, default="True")
parser.add_argument("--tbsize", type=int, default=128)
parser.add_argument("--adv", type=str, default="none", choices=["none", "fgsm", "pgd", "ball"])
parser.add_argument("--alpha", type=float, default=2.)
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--repeat", type=int, default=5)

args = parser.parse_args()


if __name__ == "__main__" :
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.save = os.path.join("experiments",args.save)
    if os.path.exists(args.save) and args.load == "none" :
        raise NameError("previous experiment '{}' already exists!".format(args.save))
    os.makedirs(args.save)

    logger = init_logger(logpath=args.save, experiment_name="logs-"+args.model)
    logger.info(args)

    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, train_eval_loader = get_cifar10_loaders(data_aug=True, batch_size=args.tbsize)

    model = cifar_model(args.model, layers=args.block, norm_type=args.norm)
    logger.info(model)
    model.to(args.device)

    loader = {"train_loader": train_loader, "train_eval_loader": train_eval_loader, "test_loader": test_loader}
    if args.opt =="sgd" :
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay, momentum=0.9, nesterov=args.nesterov)
        if args.adv == "none" :
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,100,140], gamma=0.1)
            if args.epochs <= 100 :
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)
        elif args.lr < 0.1 :
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120,160,180], gamma=0.1)
        else :
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,140,180], gamma=0.1)
    elif args.opt == "adam" :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.,0.9))
        scheduler = None

    adv_train = args.adv if args.adv != "none" else None
    model = trainer(model, logger, loader, args, "cifar10", optimizer, scheduler, adv_train=adv_train)

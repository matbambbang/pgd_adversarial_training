import os
import argparse
import time
import pickle
import torch
import torch.nn as nn
import torchvision
import numpy as np
from dataloader import get_mnist_loaders, inf_generator
from utils import init_logger, RunningAverageMeter, accuracy, one_hot
from adversarial import AttackBase, FGSM, LinfPGD, EpsilonAdversary

# Additional adversarial library
from model import BinarizeWrapper
import foolbox
import tqdm
from torch.utils.tensorboard import SummaryWriter

def adv_train_module(attack, model, data_type, iters, device, alpha=None, repeat=None) :
    norm = True if data_type != "mnist" else False
    bound = 0.3 if data_type == "mnist" else 8/255
    step = alpha or 2/255
    random_start=False if attack != "pgd" else True
    if attack == "pgd" :
        random_start = True
    else :
        random_start = False
    repeat_num = repeat or 5
    stats = (repeat_num,)

    if attack == None :
        adv = AttackBase(norm=norm, device=device)
    elif attack == "pgd" :
        adv = LinfPGD(model, bound=bound, step=step, iters=iters, norm=norm, random_start=random_start, device=device)
    elif attack == "fgsm" :
        adv = FGSM(model, bound=bound, norm=norm, random_start=random_start, device=device)
    elif attack == "ball" :
        adv = EpsilonAdversary(model, epsilon=bound, repeat=repeat_num, norm=norm, device=device)

    return adv, stats


def trainer(model, logger, loader, args, data="mnist", optimizer=None, scheduler=None, adv_train=None, tboard=True, **kwargs) :
    logger.info("="*80)
    logger.info("Train Info")
    logger.info("Model : {}".format(args.model))
    logger.info("Number of blocks : {}".format(args.block))
    logger.info("Number of parameters : {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    start_time = time.time()
    best_acc = 0.
    device = args.device

    try :
        criterion = model.loss().to(args.device)
    except :
        criterion = nn.NLLLoss().to(args.device)
    logger.info("Criterion : {}".format(criterion.__class__.__name__))
    logger.info("Adversarial Training : {}".format(adv_train))
    logger.info("="*80)
    data_gen = inf_generator(loader['train_loader'])
    batches_per_epoch = len(loader['train_loader'])

    writer = SummaryWriter(log_dir=os.path.join(args.save, args.model+"_"+str(args.block)+"_"+str(adv_train)))
    if data == "mnist" :
        dummy_input = torch.rand(1,1,28,28).to(device)
    else :
        dummy_input = torch.rand(1,3,32,32).to(device)
    writer.add_graph(model, dummy_input)

    best_acc = 0.
    best_loss = 1000.
    best_acc_epoch = 0
    best_loss_epoch = 0
    batch_time_meter = RunningAverageMeter()
    end_time = time.time()
    if args.hist :
        hist_dict = dict()
    args.alpha /= 255
    adv, stats = adv_train_module(adv_train, model, data, args.iters, args.device, args.alpha, args.repeat)

    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_acc.pt"))
    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_loss.pt"))
    for itr in tqdm.tqdm(range(args.epochs * batches_per_epoch)) :
        
        if itr % batches_per_epoch == 0 and scheduler is not None :
            scheduler.step()

        model.train()
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(args.device)
        y = y.to(args.device)

        if adv_train is not None :
            x = adv.perturb(x,y,device=args.device)
            if adv_train == "ball" :
                y = torch.cat([y for _ in range(stats[0])])
        model.zero_grad()
        logits = model(x)
        loss = criterion(logits,y)

        loss.backward()
        optimizer.step()
        
        batch_time_meter.update(time.time() - end_time)
        end_time = time.time()
        writer.add_scalar("train_loss", loss.cpu().detach(), itr)

        if itr % batches_per_epoch == 0 :
            image = adv.inverse_normalize(x.cpu())
            image = torchvision.utils.make_grid(image, scale_each=False)
            writer.add_image("train_image", image, int(itr // batches_per_epoch))
            model.eval()
            with torch.no_grad() :
                train_acc, train_loss = accuracy(model, dataset_loader=loader['train_eval_loader'], device=args.device, criterion=criterion)
                val_acc, val_loss = accuracy(model, dataset_loader=loader['test_loader'], device=args.device, criterion=criterion)
                if val_acc >= best_acc :
                    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_acc.pt"))
                    best_acc = val_acc
                    best_acc_epoch = int(itr // batches_per_epoch)
                if val_loss <= best_loss :
                    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_loss.pt"))
                    best_loss = val_loss
                    best_loss_epoch = int(itr // batches_per_epoch)
                writer.add_scalar("train_loss_epoch", train_loss, int(itr // batches_per_epoch))
                writer.add_scalar("train_acc", train_acc, int(itr // batches_per_epoch))
                writer.add_scalar("validation_loss_epoch", val_loss, int(itr // batches_per_epoch))
                writer.add_scalar("validation_acc", val_acc, int(itr // batches_per_epoch))
                logger.info(
                        "Epoch {:03d} | Time {:.3f} ({:.3f}) | Train loss {:.4f} | Validation loss {:.4f} | Train Acc {:.4f} | Validation Acc {:.4f}".format(
                            int(itr // batches_per_epoch), batch_time_meter.val, batch_time_meter.avg, train_loss, val_loss, train_acc, val_acc
                            )
                        )
            torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_final.pt"))

    torch.save({"state_dict": model.state_dict(), "args": args}, os.path.join(args.save, "model_final.pt"))
    if args.hist :
        with open(os.path.join(args.save,"history.json"),"w") as f :
            json.dump(hist_dict,f)

    logger.info("="*80)
    logger.info("Required Time : {:03d} minute {:.2f} seconds".format(int((time.time()-start_time) // 60), (time.time()-start_time) % 60))
    logger.info("Best Acc Epoch : {:03d}".format(best_acc_epoch))
    logger.info("Best Validation Accuracy : {:.4f}".format(best_acc))
    logger.info("Best loss Epoch : {:03d}".format(best_loss_epoch))
    logger.info("Best Validation loss : {:.4f}".format(best_loss))
    logger.info("Train end")
    logger.info("="*80)
    writer.close()

    return model

def test(model, target_loader, device, **kwargs) :
    model.eval()
    with torch.no_grad() :
        eval_acc, eval_loss = accuracy(model, dataset_loader=target_loader, device=device, criterion = model.loss().to(device))
    return eval_acc, eval_loss


import os
import pathlib
import random
import shutil
import time
import datetime
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from utils.logging import *
from utils.net_utils import save_checkpoint, get_lr, LabelSmoothing, set_spatial_mask
from utils.schedulers import get_policy
from args import args
from trainer import train, validate

import math
import data
import models

import pdb



def main():
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    #set_model(model)

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    ensure_path(log_base_dir)
    ensure_path(ckpt_base_dir)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logger = get_logger(os.path.join(log_base_dir, 'logger'+now+'.log'))

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)
        
    # Evaulation of a model
    if args.evaluate:
        checkpoint = torch.load(args.evaluate_model_link)
        model.load_state_dict(checkpoint["state_dict"])        
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, logger, epoch=args.start_epoch
        )
        return

    epoch_time = AverageMeter("epoch_time", ":.4f")
    validation_time = AverageMeter("validation_time", ":.4f")
    train_time = AverageMeter("train_time", ":.4f")

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        #get_distribution(model, epoch, logger)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, logger
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, logger, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)

        end_epoch = time.time()

def set_gpu(args, model):
    # DataParallel will divide and allocate batch_size to all available GPUs
    print(f"=> Parallelizing on {args.gpus} gpus")
    torch.cuda.set_device(args.gpus[0])
    args.gpu = args.gpus[0]
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda(
        args.gpus[0]
    )
    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cuda:1')
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)
    return dataset


def get_model(args):
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](deploy=args.deploy)
    if args.pretrain:
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint, strict=False) 
        set_spatial_mask(model, args)
    print(f"=> Num model params {sum(p.numel() for p in model.parameters())}")
    return model

def adjust_learning_rate(optimizer, epoch):
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    if epoch < args.warmup_length:
        lr = args.lr * (epoch + 1) / args.warmup_length
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('para_lr: {}'.format(lr))

def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            pass #print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            pass #print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = model.parameters()
        optimizer = torch.optim.SGD(
            parameters,
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    return optimizer

def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()

def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"../runs/{config}/{args.name}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir

if __name__ == "__main__":
    main()

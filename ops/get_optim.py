import torch

import utils
    
def get_optimiser(args,model):
    custom_keys_weight_decay = []
    if args.optim.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.optim.bias_weight_decay))
    if args.optim.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.optim.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.optim.weight_decay,
        norm_weight_decay=args.optim.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    opt_name = args.optim.name.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.optim.lr,
            momentum=args.optim.momentum,
            weight_decay=args.optim.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.optim.lr, momentum=args.optim.momentum, weight_decay=args.optim.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.optim.lr, weight_decay=args.optim.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
        
    return optimizer

def get_scheduler(args,optimizer):

    lr_scheduler = args.schedule.name.lower()
    if lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif lr_scheduler =='multistep':
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.schedule.milestones, gamma=args.schedule.lr_gamma)
    elif lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.schedule.epochs - args.schedule.lr_warmup_epochs, eta_min=args.schedule.lr_min
        )
    elif lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.schedule.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.schedule.lr_warmup_epochs > 0:
        if args.schedule.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.schedule.lr_warmup_decay, total_iters=args.schedule.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.schedule.lr_warmup_decay, total_iters=args.schedule.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.schedule.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.schedule.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
    
    return lr_scheduler
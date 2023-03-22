try:
    from apex import amp
    import apex
except ImportError:
    amp = None
import torch
import utils
import os


def select_training_param(model):
#     print(model)
    for v in model.parameters():
        v.requires_grad = False
    try:
        torch.nn.init.xavier_uniform_(model.linear.weight)
        model.linear.weight.requires_grad = True
        try:
            model.linear.bias.data.fill_(0.01)
            model.linear.bias.requires_grad = True
        except AttributeError:
            pass
    except AttributeError:
        torch.nn.init.xavier_uniform_(model.fc.weight)
        try:
            model.fc.bias.requires_grad = True
            model.fc.bias.data.fill_(0.01)
        except AttributeError:
            pass
        model.fc.weight.requires_grad = True
        

    return model


def finetune_places(model):
#     print(model)
    for v in model.parameters():
        v.requires_grad = False
    
    torch.nn.init.xavier_uniform_(model.fc.weight)
    try:
        model.fc.bias.requires_grad = True
    except AttributeError:
        pass
    model.fc.weight.requires_grad = True
    
    for v in model.layer4.parameters():
        v.requires_grad = True
        

    return model



def load_model(args,model,optimizer,lr_scheduler):
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )
    if args.experiment.decoup:
        model = select_training_param(model)

    if args.experiment.fn_places is True:
        model = finetune_places(model)


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.experiment.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.experiment.model_ema_steps / args.schedule.epochs
        alpha = 1.0 - args.experiment.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=torch.device('cuda'), decay=1.0 - alpha)
    
    
    if args.resume:
        resume_file = os.path.join(os.getenv('owd'),"./output",args.experiment.name, f"last.pth")
        checkpoint = torch.load(resume_file, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.experiment.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
        elif args.apex:
            amp.load_state_dict(checkpoint["amp"])

    if args.load_from:
        resume_file = os.path.join(os.getenv('owd'),"./output",args.experiment.name, args.load_from)
        checkpoint = torch.load(resume_file, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"],strict=False)
        
    return model,model_ema,scaler
    

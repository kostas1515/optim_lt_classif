import hydra
from omegaconf import DictConfig
import torch
import os
import utils
from datasets.loader import load_data, load_loaders
from models import initialise_model
from ops import get_optim, load_model
import time
from engine import trainer,tester
import datetime


@hydra.main(config_path="hydra",config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ['owd'] = hydra.utils.get_original_cwd()
    utils.init_distributed_mode(cfg)
    device = torch.device('cuda')

    if cfg.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset, dataset_test, train_sampler, test_sampler = load_data(cfg)


    data_loader, data_loader_test = load_loaders(cfg,dataset, dataset_test, train_sampler, test_sampler)
    

    print("Creating model")
    model = initialise_model.get_model(cfg)
    model.to(device)

    if cfg.distributed and cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = initialise_model.get_criterion(cfg,dataset)

    optimizer = get_optim.get_optimiser(cfg,model)

    
    lr_scheduler = get_optim.get_scheduler(cfg,optimizer)
    
    model,model_ema,scaler = load_model.load_model(cfg,model,optimizer,lr_scheduler)

    if cfg.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            tester.evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            tester.evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    best_acc = 0
    for epoch in range(cfg.experiment.start_epoch, cfg.schedule.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        trainer.train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, cfg, model_ema, scaler)
        lr_scheduler.step()
        acc = tester.evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            acc=tester.evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if cfg.experiment.name:
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": cfg,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            elif cfg.apex:
                checkpoint["amp"] = amp.state_dict()
            if acc>best_acc:
                best_acc = acc
                utils.save_on_master(checkpoint, os.path.join(os.getenv('owd'),"./output",cfg.experiment.name, f"best.pth"))
            utils.save_on_master(checkpoint, os.path.join(os.getenv('owd'),"./output",cfg.experiment.name, f"last.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    return best_acc
  
    
if __name__=='__main__':
    main()
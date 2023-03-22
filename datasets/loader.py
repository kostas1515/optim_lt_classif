import os
import time
import torch
import torch.utils.data
import torchvision


from datasets import presets
import utils
from ops.sampler import RASampler
from ops import transforms 
from datasets import imbalanced_dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

from catalyst.data import  BalanceClassSampler,DistributedSamplerWrapper


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(cfg):
    cwd = os.getenv('owd')
    root = os.path.join(cwd,cfg.dataset.root)
    traindir = os.path.join(cwd,cfg.dataset.root, "train")
    valdir = os.path.join(cwd,cfg.dataset.root, "val")
    
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        cfg.dataset.val_resize_size,
        cfg.dataset.val_crop_size,
        cfg.dataset.train_crop_size,
    )
    interpolation = InterpolationMode(cfg.augmentations.interpolation)

    
    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cfg.dataset.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(cfg.augmentations, "auto_augment", None)
        random_erase_prob = getattr(cfg.augmentations, "random_erase", 0.0)
        ra_magnitude = cfg.augmentations.ra_magnitude
        augmix_severity = cfg.augmentations.augmix_severity
        train_transform = presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                )
        num_classes = cfg.dataset.num_classes
        if cfg.dataset.dset_name == 'ImageNet':
            dataset = torchvision.datasets.ImageFolder(
                traindir,train_transform,
            )
        elif cfg.dataset.dset_name.startswith('cifar') is True:
            dataset, dataset_test = imbalanced_dataset.load_cifar(cfg)
        else:
            dataset = imbalanced_dataset.LT_Dataset(root,os.path.join(cwd,cfg.dataset.train_txt),num_classes, transform=train_transform)
        
        if cfg.dataset.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cfg.dataset.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if cfg.weights and cfg.test_only:
            weights = torchvision.models.get_weight(cfg.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )
        if cfg.dataset.dset_name == 'ImageNet':
            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
        elif cfg.dataset.dset_name.startswith('cifar') is True:
            pass # test dataset already loaded
        else:
            dataset_test = imbalanced_dataset.LT_Dataset_Eval(root, os.path.join(cwd,cfg.dataset.eval_txt),dataset.class_map, num_classes, transform=preprocessing)
            
        if cfg.dataset.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if cfg.distributed:
        if hasattr(cfg.experiment, "ra_sampler") and cfg.experiment.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=cfg.experiment.ra_reps)
            
        else:
            if cfg.experiment.sampler=='random':
                if cfg.hpopt is True:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=cfg.gpu,rank=cfg.rank)
                else:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                train_labels = dataset.targets
                balanced_sampler = BalanceClassSampler(train_labels,mode=cfg.experiment.sampler)
                train_sampler= DistributedSamplerWrapper(balanced_sampler)
        if cfg.hpopt is True:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test,num_replicas=cfg.gpu,rank=cfg.rank,shuffle=False)
        else:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if cfg.experiment.sampler=='random':
            train_sampler = torch.utils.data.RandomSampler(dataset)
        else:
            train_labels = dataset.targets
            train_sampler = BalanceClassSampler(train_labels,mode=cfg.experiment.sampler)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler



def load_loaders(cfg,dataset,dataset_test,train_sampler,test_sampler):
    collate_fn = None
    num_classes = cfg.dataset.num_classes
    mixup_transforms = []
    if cfg.augmentations.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=cfg.augmentations.mixup_alpha))
    if cfg.augmentations.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=cfg.augmentations.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))
    if cfg.hpopt is True:
        mp_context='fork'
    else:
        mp_context=None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        multiprocessing_context=mp_context
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.batch_size, sampler=test_sampler, num_workers=cfg.workers, pin_memory=True,multiprocessing_context=mp_context
    )
    
    return data_loader,data_loader_test
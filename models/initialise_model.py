import torch
import torch.nn as nn
import os
from models.vit_pytorch import SimpleViT
from models import resnet_pytorch
from models import resnet_cifar
from models import custom

def _mismatched_classifier(model,pretrained):
    classifier_name, old_classifier = model._modules.popitem()
    classifier_input_size = old_classifier[1].in_features
    
    pretrained_classifier = nn.Sequential(
                nn.LayerNorm(classifier_input_size),
                nn.Linear(classifier_input_size, 1000)
            )
    model.add_module(classifier_name, pretrained_classifier)
    state_dict = torch.load(pretrained, map_location='cpu')
    model.load_state_dict(state_dict['model'],strict=False)

    classifier_name, new_classifier = model._modules.popitem()
    model.add_module(classifier_name, old_classifier)
    return model

def get_model(args):
    if args.pretrained is not "None":
        cwd = os.getenv('owd')
        pretrained = os.path.join(cwd,args.pretrained)
    else:
        pretrained = args.pretrained 
    num_classes=args.dataset.num_classes
    if args.model.name.endswith('vit') is True:
        if args.model.name == 'simple_vit':
            model = SimpleViT(
                    image_size = args.dataset.train_crop_size,
                    patch_size = args.model.patch_size,
                    num_classes = num_classes,
                    dim = args.model.dim,
                    depth = args.model.depth,
                    heads = args.model.heads,
                    mlp_dim = args.model.mlp_dim,
                    attention=args.model.attention,
                    use_norm=args.model.classif_norm)
        if args.pretrained is not None:
            if num_classes!=1000:
                model = _mismatched_classifier(model,args.pretrained)
    else:
        try:
            # model = torchvision.models.__dict__[args.model](pretrained=args.pretrained,num_classes=num_classes)
            print(f'resnet_pytorch.{args.model.name}(num_classes={num_classes},use_norm="{args.model.classif_norm}",use_gumbel={args.model.use_gumbel_se},use_gumbel_cb={args.model.use_gumbel_cb},pretrained="{pretrained}")')
            model = eval(f'resnet_pytorch.{args.model.name}(num_classes={num_classes},use_norm="{args.model.classif_norm}",use_gumbel={args.model.use_gumbel_se},use_gumbel_cb={args.model.use_gumbel_cb},pretrained="{pretrained}")')
        except AttributeError:
            #model does not exist in pytorch load it from resnet_cifar
            model = eval(f'resnet_cifar.{args.model.name}(num_classes={num_classes},use_norm="{args.model.classif_norm}",use_gumbel={args.model.use_gumbel_se},use_gumbel_cb={args.model.use_gumbel_cb})')
            
    model = initialise_classifier(args,model,num_classes)
    
    return model

def get_weights(dataset):
    per_cls_weights = torch.tensor(dataset.get_cls_num_list(),device='cuda')
    per_cls_weights = per_cls_weights.sum()/per_cls_weights
    return per_cls_weights

def get_criterion(args,dataset):
    if args.criterion.deffered:
        weight=get_weights(dataset)
    else:
        weight=None
    if args.criterion.name =='ce':
        return torch.nn.CrossEntropyLoss(label_smoothing=args.criterion.label_smoothing,weight=weight)
    elif args.criterion.name =='gce':
        return custom.BCE(label_smoothing=args.criterion.label_smoothing,use_gumbel=True,weight=weight,reduction=args.criterion.reduction)
    elif args.criterion.name =='iif':
        return custom.IIFLoss(dataset,weight=weight,variant=args.criterion.iif,label_smoothing=args.criterion.label_smoothing)
    elif args.criterion.name =='bce':
        return custom.BCE(label_smoothing=args.criterion.label_smoothing,reduction=args.criterion.reduction,weight=weight,)
        


def initialise_classifier(args,model,num_classes):
    num_classes = torch.tensor([num_classes])
    if args.criterion.name == 'gce':
        if args.model.name.endswith('vit') is True:
            torch.nn.init.normal_(model.linear_head[-1].weight.data,0.0,0.001)
            try:
                torch.nn.init.constant_(model.linear_head[-1].bias.data,-2.0)
            except AttributeError:
                print('no bias in classifier head')
                pass
        else:
            if args.dataset.dset_name.startswith('cifar'):
                torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
            else:
                torch.nn.init.normal_(model.fc.weight.data,0.0,0.001)
            try:
                if args.dataset.dset_name.startswith('cifar'):
                    torch.nn.init.constant_(model.linear.bias.data,-torch.log(torch.log(num_classes)).item())
                else:
                    torch.nn.init.constant_(model.fc.bias.data,-torch.log(torch.log(num_classes)).item())
            except AttributeError:
                print('no bias in classifier head')
                pass
    elif args.criterion.name == 'bce':
        if args.model.name.endswith('vit') is True:
            torch.nn.init.normal_(model.linear_head[-1].weight.data,0.0,0.001)
            try:
                torch.nn.init.constant_(model.linear_head[-1].bias.data,-6.5)
            except AttributeError:
                print('no bias in classifier head')
                pass
        else:
            if args.dataset.dset_name.startswith('cifar'):
                torch.nn.init.normal_(model.linear.weight.data,0.0,0.001)
            else:
                torch.nn.init.normal_(model.fc.weight.data,0.0,0.001)
            try:
                if args.dataset.dset_name.startswith('cifar'):
                    torch.nn.init.constant_(model.linear.bias.data,-6.0)
                else:
                    torch.nn.init.constant_(model.fc.bias.data,-6.0)
            except AttributeError:
                print('no bias in classifier head')
                pass
    return model
        
    

import torch
from torch import nn
from torchvision import models
import os


def save_checkpoint(model, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint = {
        'arch': model.arch,
        'classifier': list(model.classifier.children()),
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    arch = checkpoint['arch']
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)

    model.classifier = nn.Sequential(*checkpoint['classifier'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.arch = arch

    return model

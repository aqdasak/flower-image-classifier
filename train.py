#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Aqdas Ahmad Khan
# DATE CREATED: 28 Feb 2023
# REVISED DATE:
# PURPOSE:
##

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from workspace_utils import active_session
from get_input_args import get_train_input_args
from utils import save_checkpoint


def get_image_datasets(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        'validation': transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        'test': transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'validation': datasets.ImageFolder(
            valid_dir, transform=data_transforms['validation']
        ),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    return image_datasets


def get_dataloaders(image_datasets):
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=64, shuffle=True
        ),
        'validation': torch.utils.data.DataLoader(
            image_datasets['validation'], batch_size=64, shuffle=True
        ),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=64, shuffle=True
        ),
    }

    return dataloaders


def get_model(arch, hidden_units):
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print('Only vgg11 and vgg13 are available right now.')
        exit()

    # Freezing feature parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1),
    )

    model.arch = arch
    return model


def train_model(
    model, dataloaders, epochs=5, learning_rate=0.003, print_every=20, gpu=False
):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    trainloader = dataloaders['train']
    validationloader = dataloaders['validation']

    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)

    running_loss = 0
    validation_loss = 0
    accuracy = 0
    with active_session():
        for e in range(1, epochs + 1):
            model.train()
            for step, (images, labels) in enumerate(trainloader, start=1):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if step % print_every == 0:
                    with torch.no_grad():
                        model.eval()

                        for images, labels in validationloader:
                            images, labels = images.to(device), labels.to(device)

                            log_ps = model.forward(images)
                            loss = criterion(log_ps, labels)

                            validation_loss += loss.item()

                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(
                                equals.type(torch.FloatTensor)
                            ).item()

                        model.train()

                    print(
                        f"Epoch {e}/{epochs}({step/print_every}) "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validationloader):.3f}.."
                    )
                    running_loss = 0
                    validation_loss = 0
                    accuracy = 0

    return model


def validate_model(model, dataloaders, gpu=False):
    testloader = dataloaders['test']
    criterion = nn.NLLLoss()

    device = torch.device("cuda" if gpu else "cpu")
    model = model.to(device)
    with torch.no_grad():
        model.eval()

        test_loss = 0
        accuracy = 0
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)

            test_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(
        f"Test loss: {test_loss/len(testloader):.3f}.. "
        f"Test accuracy: {accuracy/len(testloader):.3f}.."
    )


def main():
    arg = get_train_input_args()

    image_datasets = get_image_datasets(arg.data_dir)
    dataloaders = get_dataloaders(image_datasets)

    model = get_model(arg.arch, arg.hidden_units)
    model.class_to_idx = image_datasets['train'].class_to_idx

    model = train_model(
        model, dataloaders, arg.epochs, arg.learning_rate, print_every=20, gpu=arg.gpu
    )
    validate_model(model, dataloaders, arg.gpu)

    save_checkpoint(model, arg.save_dir)


if __name__ == '__main__':
    main()

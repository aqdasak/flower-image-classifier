#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Aqdas Ahmad Khan
# DATE CREATED: 28 Feb 2023
# REVISED DATE:
# PURPOSE:
##

import torch
import numpy as np
from PIL import Image
import json

from get_input_args import get_predict_input_args
from utils import load_checkpoint


def process_image(image):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''

    def resize_image(image, px):
        # Resizing image's shortest side to 256, maintaining the aspect ratio
        w, h = image.size
        if w < h:
            return image.resize((px, int(h / w * px)))
        else:
            return image.resize((int(w / h * px), px))

    def center_crop_image(image, size):
        # Center crop to size
        w, h = size

        left = int(image.size[0] / 2 - w / 2)
        upper = int(image.size[1] / 2 - h / 2)
        right = left + w
        lower = upper + h

        return image.crop((left, upper, right, lower))

    def normalize_numpy_image(np_image, means, stds):
        return (np_image - means) / stds

    image = resize_image(image, 256)
    image = center_crop_image(image, (224, 224))
    np_image = np.array(image)
    np_image = np_image / 255
    n_image = normalize_numpy_image(
        np_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image
    n_image = n_image.transpose((2, 0, 1))
    return torch.from_numpy(n_image).float()  # Returning torch tensor


def predict(image_path, model, topk=1, gpu=False):
    '''Predict the class (or classes) of an image using a trained deep learning model.'''

    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)

    model.eval()
    with Image.open(image_path) as image:
        image = process_image(image)
        # Adding 1 additional dimension as it was during training as batches
        image = image.view((1, *image.shape))

        image = image.type(torch.cuda.FloatTensor if gpu else torch.FloatTensor)
        with torch.no_grad():
            log_ps = model.forward(image)
            ps = torch.exp(log_ps)
            top_ps, top_classes = ps.topk(topk, dim=1)
            return top_ps.view(topk), top_classes.view(topk)


def main():
    arg = get_predict_input_args()
    model = load_checkpoint(arg.checkpoint)

    probs, classes = predict(arg.image_path, model, arg.top_k, arg.gpu)
    probs, classes = probs.to('cpu').numpy(), classes.to('cpu').numpy()

    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    classes = [idx_to_class[idx] for idx in classes]

    if arg.category_names != '':
        with open(arg.category_names) as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[category] for category in classes]

    for name, prob in zip(classes, probs):
        print(f'{name}: {prob:.3f}')


if __name__ == '__main__':
    main()

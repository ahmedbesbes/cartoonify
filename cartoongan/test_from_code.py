import time
import os

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from network.Transformer import Transformer


def transform(models, style, input, load_size=450, gpu=-1):
    model = models[style]

    if gpu > -1:
        model.cuda()
    else:
        model.float()

    input_image = Image.open(input).convert("RGB")
    h, w = input_image.size

    ratio = h * 1.0 / w

    if ratio > 1:
        h = load_size
        w = int(h * 1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)

    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)

    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    input_image = -1 + 2 * input_image
    if gpu > -1:
        input_image = Variable(input_image).cuda()
    else:
        input_image = Variable(input_image).float()

    t0 = time.time()
    print("input shape", input_image.shape)
    with torch.no_grad():
        output_image = model(input_image)[0]
    print(f"inference time took {time.time() - t0} s")

    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    output_image = output_image.numpy()

    output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)
    output_image = Image.fromarray(output_image)

    return output_image

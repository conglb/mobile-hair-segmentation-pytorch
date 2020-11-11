import cv2
import torch
from model.model import MobileHairNet
from config.config import get_config
import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt


def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

if __name__ == "__main__":

    pretrained = glob(os.path.join("checkpoints", "MobileHairNet_epoch-141.pth"))[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MobileHairNet().to(device)
    net.load_state_dict(torch.load(pretrained, map_location=device))

    for i, param in enumerate(net.parameters()):
        print(param.shape)
        if i==14:
            plot_kernels(param.detach().numpy())
    
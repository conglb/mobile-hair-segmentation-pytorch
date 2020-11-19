import torch
from model.model import MobileHairNet
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils.custom_transfrom import UnNormalize
from loss.loss import iou_loss

class Tester:
    def __init__(self, config, dataloader):
        self.batch_size = config.batch_size
        self.config = config
        self.model_path = config.checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.num_classes = config.num_classes
        self.num_test = config.num_test
        self.sample_dir = config.sample_dir
        self.epoch = config.epoch
        self.build_model()

    def build_model(self):
        self.net = MobileHairNet()
        self.net.to(self.device)
        self.load_model()

    def load_model(self):
        print("[*] [TEST] Load checkpoint in ", str(self.model_path))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.listdir(self.model_path):
            print("[!] [TEST] No checkpoint in ", str(self.model_path))
            return

        model_path = os.path.join(self.model_path, f"MobileHairNet_epoch-{self.epoch}.pth")
        model = glob(model_path)
        model.sort()
        if not model:
            raise Exception(f"[!] [TEST] No Checkpoint in {model_path}")

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print(f"[*] [TEST] Load Model from {model[-1]}: ")

    def test(self):
        #unnormal = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        total_iou = 0
        print("[TEST] Number of image in testset: {}".format(self.data_loader.batch_size * self.data_loader.__len__()))
        for step, (image, gray, mask) in enumerate(self.data_loader):
            #image = unnormal(image.to(self.device))
            mask = mask.to(self.device).repeat_interleave(3, 1)
            result = self.net(image)
            '''
            if step == 0:
                print(image.shape)
                print(gray.shape)
                print(mask.shape)
            '''
            iou = iou_loss(result, mask)
            print(iou)
            total_iou += iou

            argmax = torch.argmax(result, dim=1).unsqueeze(dim=1)
            result = result[:, 1, :, :].unsqueeze(dim=1)
            result = result * argmax
            result = result.repeat_interleave(3, 1)
            #torch.cat([image, result, mask])
            #save_image(torch.cat([image, result, mask]), os.path.join(self.sample_dir, f"{step}.png"))
            #print('[*] [TEST] Saved sample images')
            
        print(total_iou / len(self.data_loader))

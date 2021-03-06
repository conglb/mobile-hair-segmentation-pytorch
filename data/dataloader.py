#
#
# data folder 구조
# (data_folder) / original
# (data_folder) / mask
# (data_folder) / ...
# (data_folder) / ...
#
#
import os

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import random
from data.figaro import *
from data.lfw import *


def transform(image, mask, image_size=224):
    # Resize
    resized_num = int(random.random() * image_size)
    resize = transforms.Resize(size=(image_size + resized_num, image_size + resized_num))
    image = resize(image)
    mask = resize(mask)

    # num_pad = int(random.random() * image_size)
    # image = TF.pad(image, num_pad, padding_mode='edge')
    # mask = TF.pad(mask, num_pad)

    # # Random crop
    # i, j, h, w = transforms.RandomCrop.get_params(
    #     image, output_size=(image_size, image_size))
    # image = TF.crop(image, i, j, h, w)
    # mask = TF.crop(mask, i, j, h, w)


    # # Random horizontal flipping
    # if random.random() > 0.5:
    #     image = TF.hflip(image)
    #     mask = TF.hflip(mask)
    #
    # # Random vertical flipping
    # if random.random() > 0.5:
    #     image = TF.vflip(image)
    #     mask = TF.vflip(mask)

    resize = transforms.Resize(size=(image_size, image_size))
    image = resize(image)
    mask = resize(mask)

    # Make gray scale image
    gray_image = TF.to_grayscale(image)

    # Transform to tensor
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    gray_image = TF.to_tensor(gray_image)

    # Normalize Data
    image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return image, gray_image, mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, image_size):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception(f"[!] {self.data_folder} not exists.")

        self.objects_path = []
        self.image_name = os.listdir(os.path.join(data_folder, "images"))
        if len(self.image_name) == 0:
            raise Exception(f"No image found in {self.image_name}")
        for p in os.listdir(data_folder):
            if p == "images":
                continue
            self.objects_path.append(os.path.join(data_folder, p))

        self.image_size = image_size

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_folder, 'images', self.image_name[index])).convert('RGB')
        mask = Image.open(os.path.join(self.data_folder, 'masks', self.image_name[index]))

        image, gray_image, mask = transform(image, mask)


        return image, gray_image, mask

    def __len__(self):
        return len(self.image_name)

def get_joint_transforms():
    return transforms.Compose(
        [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ]
    )

class AllDataset(torch.utils.data.ConcatDataset):
    
    def __init__(self, data_folder, image_size):
        super(AllDataset, self).__init__([LfwDataset(os.path.join(data_folder,'Lfw'), joint_transforms=get_joint_transforms()), FigaroDataset(os.path.join(data_folder,'Figaro1k'), joint_transforms=get_joint_transforms())])
            

def get_loader(data_folder, batch_size, image_size, shuffle, num_workers):
    dataset = FigaroDataset(os.path.join(data_folder,'Figaro1k'), joint_transforms=get_joint_transforms())
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader



if __name__ == "__main__":
    from figaro import FigaroDataset
    from lfw import LfwDataset
    loader = get_loader('./dataset', 10, 224,  shuffle=False, num_workers=1)
    for i, batch in enumerate(loader):
        img, gray, mask = batch 
        trans = transforms.ToPILImage()
        gray = trans(gray[0])
        gray.save('./dataset/{}-gray.jpg'.format(i))


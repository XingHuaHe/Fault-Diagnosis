# systerm packages.
import os
import sys
from typing import Type, Tuple
# Extend packages.
import numpy as np
import scipy.io as scio
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SignalDataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str, transforms: transforms = None, mode: str = "Raw", train: bool = True) -> None:
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.mode = mode
        self.train = train
        self.data_info = self.get_data_info(self.data_dir, self.label_dir, self.mode)

    def __getitem__(self, index: int) -> tuple:
        path_img, label = self.data_info[index]
        if self.mode == "DogCat":
            img = Image.open(path_img).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            else:
                img = transforms.ToTensor(img)
            return img, label
        elif self.mode == "Raw":
            img = np.array([path_img])
            img_T = img.transpose()
            img = torch.unsqueeze(torch.Tensor(img_T.dot(img)), 0)
            if self.transforms is not None:
                img = self.transforms(img)
            return img, label

    def __len__(self) -> int:
        return len(self.data_info)

    def get_data_info(self, data_dir, label_dir, mode):
        if data_dir != None and mode == "DogCat":
            # Dog and Cat classification work.
            classDirs = os.listdir(data_dir)
            dog_cat_path = [os.path.join(data_dir, i) for i in classDirs]

            data_info = list()
            for i in range(len(dog_cat_path)):
                if dog_cat_path[i].split('/')[-1] == "cats":
                    img_names = os.listdir(dog_cat_path[i])
                    for name in img_names:
                        path_img = os.path.join(dog_cat_path[i], name)
                        # 0 represented cat.
                        data_info.append((path_img, 0))
                elif dog_cat_path[i].split('/')[-1] == "dogs":
                    img_names = os.listdir(dog_cat_path[i])
                    for name in img_names:
                        path_img = os.path.join(dog_cat_path[i], name)
                        # 1 represented cat.
                        data_info.append((path_img, 1))
            return data_info

        elif data_dir != None and mode == "Raw":
            signals = scio.loadmat(data_dir)
            labels = scio.loadmat(label_dir)
            if self.train:
                # load datas.
                signals = signals['train_signals']
                # load labels.
                labels = labels["train_labels"]
            else:
                # load datas.
                signals = signals['test_signals']
                # load labels.
                labels = labels["test_labels"]        
            # construct tupe.
            data_info = list()
            for i in range(len(signals)):
                data_info.append((signals[i], int(labels[i][0]-1)))
            return data_info
        else:
            return None

import numpy as np
import pandas as pd
import torchvision as tv
import torch
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import gray2rgb


# Precomputed for normalization
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame, mode: str):
        self.data = data
        self.mode = mode

        # Perform data augmentation
        self._transform_train = tv.transforms.Compose([
            tv.transforms.ToPILImage(), 
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip(),
            tv.transforms.RandomAffine(degrees=(-3, 3), translate=(0.02, 0.02)),
            tv.transforms.RandomResizedCrop((300, 300), scale=(0.98, 1.0), ratio=(1.0, 1.0)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])
        
        self._transform_val = tv.transforms.Compose([
            tv.transforms.ToPILImage(), 
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        gray_image = imread(self.data.at[index, "filename"])
        rgb_image = torch.from_numpy(np.transpose(gray2rgb(gray_image), (2, 0, 1)))
        labels = torch.tensor([self.data.at[index, "crack"], self.data.at[index, "inactive"]])
        trans_image = self._transform_train(rgb_image) if self.mode == "train" else self._transform_val(rgb_image)
        return trans_image, labels.float()

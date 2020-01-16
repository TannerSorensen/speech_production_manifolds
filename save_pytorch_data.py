#!/usr/bin/env python3

import os
from glob import glob

import numpy as np
import pandas as pd
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader

class VocalTractDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # get list of filenames
        self.fname = glob(os.path.join(self.root_dir, '*', 'png', '*', '*.png'))

        # read annotations
        self.annotations = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.fname)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read image
        img_name = self.fname[idx]
        image = io.imread(img_name)

        participant = img_name.split(os.path.sep)[-4]
        avi = img_name.split(os.path.sep)[-2]
        img = int(img_name.split(os.path.sep)[-1].replace(".png", ""))

        # read annotations
        phone = self.annotations[(self.annotations["participant"] == participant) & (self.annotations["avi"] == avi) & (self.annotations["img"] == img)].values
        if (phone.shape[0] > 0):
            phone = phone[0][-1]
        else:
            phone = ""
        phone = np.array([phone])
        phone = phone.astype('str')
        sample = {'image': image, 'phone': phone}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__=="__main__":

    vocal_tract_dataset = VocalTractDataset(csv_file="/home/tsorense/speech_production_manifolds/annotations/endpoints.csv",
            root_dir="/home/tsorense/speech_production_manifolds/speech_production_manifolds_data/")

    for i in range(len(vocal_tract_dataset)):
        sample = vocal_tract_dataset[i]
        print(i, sample['image'].shape, sample['phone'].shape)

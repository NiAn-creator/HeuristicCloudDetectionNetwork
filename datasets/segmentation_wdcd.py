import os
import torch.utils.data as data
import numpy as np
import torch

class wscd_train_wdcd(data.Dataset):

    def __init__(self, root, mask_dir, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root


        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]

        assert len(self.images) == len(self.masks)

    # just return the img and target in P[key]
    def __getitem__(self, index):

        # get the hyperspectral data and lables

        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        rsData = rsData.transpose(2,0,1)

        weakLable = np.load(self.masks[index])
        weakLable = np.asarray(weakLable, dtype=np.float32)
        weakLable = np.expand_dims(weakLable, axis=0)

        hedLable = np.zeros((321, 321), dtype=np.float32)
        hedLable[weakLable > 0] = 1

        img = torch.tensor(rsData)
        weak = torch.tensor(weakLable)
        hed = torch.tensor(hedLable)

        mean = torch.tensor([441.14345306, 443.68343218, 380.8605016, 307.75192367])
        mean_re = mean.view(4, 1, 1).expand((4, 250, 250))
        img = img - mean_re

        return img, weak, hed

    def __len__(self):
        return len(self.images)

class wscd_test_wdcd(data.Dataset):

    def __init__(self, root, image_set='test'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set
        voc_root = self.root


        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if not os.path.exists(image_dir):
            raise ValueError(
                'Wrong image_dir entered!')
        if not os.path.exists(mask_dir):
            raise ValueError(
                'Wrong mask_dir entered!')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]

        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):

        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData,dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)

        target = np.load(self.masks[index])
        target = np.asarray(target,dtype=np.float32)

        img = torch.Tensor(rsData)
        target = torch.Tensor(target)

        mean = torch.tensor([301.37868060, 289.88378237, 259.03378350, 295.58795369])
        mean_re = mean.view(4, 1, 1).expand((4, 250, 250))
        img = img - mean_re

        return img, target

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)

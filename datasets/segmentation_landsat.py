import os
import torch.utils.data as data
import numpy as np
import torch
import cv2
import skimage.io as skio


class wscd_train_landsat(data.Dataset):

    def __init__(self, root, mask_dir, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

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

        assert len(self.images) == len(self.masks)

    # just return the img and target in P[key]
    def __getitem__(self, index):

        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)

        weakLable = np.load(self.masks[index])
        weakLable = np.asarray(weakLable, dtype=np.float32)
        weakLable = np.expand_dims(weakLable, axis=0)

        hedLable = np.zeros((321, 321), dtype=np.float32)
        hedLable[weakLable > 0] = 1

        img = torch.tensor(rsData[:,:-1,:-1])
        weak = torch.tensor(weakLable[:,:-1,:-1])
        hed = torch.tensor(hedLable[:,:-1,:-1])

        mean = torch.tensor([20135.71790780, 19817.51281344, 18800.41467967, 19188.28884100,
                             21662.22055619, 13404.15832316, 11725.27454421, 6503.315547821])
        mean_re = mean.view(8, 1, 1).expand((8, 320, 320))
        img = img - mean_re

        return img, weak, hed

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)

class wscd_test_landsat(data.Dataset):

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

    # just return the img and target in P[key]
    def __getitem__(self, index):
        # get the hyperspectral data and lables
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData,dtype=np.float32)
        rsData = np.asarray(rsData)

        mask = np.load(self.masks[index])
        mask = np.asarray(mask, dtype=np.float32)

        img = torch.tensor(rsData[:,:-1,:-1])
        target = torch.Tensor(mask[:-1,:-1])

        if self.image_set == "test":
            mean = torch.tensor([15829.52994161,15397.82656040,14953.84301515,15423.67903396,
                                 19455.17218920,15654.74908511,12889.68558145,5413.554702360])
        elif self.image_set == "trainval":
            mean = torch.tensor([20598.81542004,20369.23671756,19613.50879086,20230.48743073,
                                 22156.93456010,14029.09723986,12666.16199967,7275.159183719])

        mean_re = mean.view(4, 1, 1).expand((4, 320, 320))
        img = img - mean_re

        return img, target

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)
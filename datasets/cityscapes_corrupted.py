import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Cityscapes_C(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', severity = 1, transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        # val_Brightness_  val_Contrast_  val_DefocusBlur_  val_Fog_  val_Frost_  val_GaussianBlur_  val_GaussianNoise_  val_GlassBlur_
        # val_ImpulseNoise_  val_JPEG_  val_Motion_Blur_  val_Saturate_  val_ShotNoise_  val_Snow_  val_Spatter_  val_SpeckleNoise_
        path_severity1= 'Cityscape_corrupted/val_Brightness_'+str(severity)
        self.images_dir1 = os.path.join(self.root, path_severity1)
        self.targets_dir1 = os.path.join(self.root, self.mode, split)

        path_severity2= 'Cityscape_corrupted/val_Contrast_'+str(severity)
        self.images_dir2 = os.path.join(self.root, path_severity2)
        self.targets_dir2 = os.path.join(self.root, self.mode, split)

        path_severity3= 'Cityscape_corrupted/val_DefocusBlur_'+str(severity)
        self.images_dir3 = os.path.join(self.root, path_severity3)
        self.targets_dir3 = os.path.join(self.root, self.mode, split)

        path_severity4= 'Cityscape_corrupted/val_Fog_'+str(severity)
        self.images_dir4 = os.path.join(self.root, path_severity4)
        self.targets_dir4 = os.path.join(self.root, self.mode, split)

        path_severity5= 'Cityscape_corrupted/val_Frost_'+str(severity)
        self.images_dir5 = os.path.join(self.root, path_severity5)
        self.targets_dir5 = os.path.join(self.root, self.mode, split)

        path_severity6= 'Cityscape_corrupted/val_GaussianNoise_'+str(severity)
        self.images_dir6 = os.path.join(self.root, path_severity6)
        self.targets_dir6 = os.path.join(self.root, self.mode, split)

        path_severity7= 'Cityscape_corrupted/val_GlassBlur_'+str(severity)
        self.images_dir7 = os.path.join(self.root, path_severity7)
        self.targets_dir7 = os.path.join(self.root, self.mode, split)

        path_severity8= 'Cityscape_corrupted/val_GaussianBlur_'+str(severity)
        self.images_dir8 = os.path.join(self.root, path_severity8)
        self.targets_dir8 = os.path.join(self.root, self.mode, split)

        path_severity9= 'Cityscape_corrupted/val_ImpulseNoise_'+str(severity)
        self.images_dir9 = os.path.join(self.root, path_severity9)
        self.targets_dir9 = os.path.join(self.root, self.mode, split)

        path_severity10= 'Cityscape_corrupted/val_JPEG_'+str(severity)
        self.images_dir10 = os.path.join(self.root, path_severity10)
        self.targets_dir10 = os.path.join(self.root, self.mode, split)

        path_severity11= 'Cityscape_corrupted/val_Motion_Blur_'+str(severity)
        self.images_dir11 = os.path.join(self.root, path_severity11)
        self.targets_dir11 = os.path.join(self.root, self.mode, split)

        path_severity12= 'Cityscape_corrupted/val_Saturate_'+str(severity)
        self.images_dir12 = os.path.join(self.root, path_severity12)
        self.targets_dir12 = os.path.join(self.root, self.mode, split)

        path_severity13= 'Cityscape_corrupted/val_ShotNoise_'+str(severity)
        self.images_dir13 = os.path.join(self.root, path_severity13)
        self.targets_dir13 = os.path.join(self.root, self.mode, split)

        path_severity14= 'Cityscape_corrupted/val_Snow_'+str(severity)
        self.images_dir14 = os.path.join(self.root, path_severity14)
        self.targets_dir14 = os.path.join(self.root, self.mode, split)

        path_severity15= 'Cityscape_corrupted/val_Spatter_'+str(severity)
        self.images_dir15 = os.path.join(self.root, path_severity15)
        self.targets_dir15 = os.path.join(self.root, self.mode, split)

        path_severity16= 'Cityscape_corrupted/val_SpeckleNoise_'+str(severity)
        self.images_dir16 = os.path.join(self.root, path_severity16)
        self.targets_dir16 = os.path.join(self.root, self.mode, split)

        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')
        print('CHECK PATH ===>', self.images_dir1)
        if not os.path.isdir(self.images_dir1) or not os.path.isdir(self.targets_dir1):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir1):
            img_dir = os.path.join(self.images_dir1, city)
            target_dir = os.path.join(self.targets_dir1, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir2):
            img_dir = os.path.join(self.images_dir2, city)
            target_dir = os.path.join(self.targets_dir2, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir3):
            img_dir = os.path.join(self.images_dir3, city)
            target_dir = os.path.join(self.targets_dir3, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir4):
            img_dir = os.path.join(self.images_dir4, city)
            target_dir = os.path.join(self.targets_dir4, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir5):
            img_dir = os.path.join(self.images_dir5, city)
            target_dir = os.path.join(self.targets_dir5, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir6):
            img_dir = os.path.join(self.images_dir6, city)
            target_dir = os.path.join(self.targets_dir6, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir7):
            img_dir = os.path.join(self.images_dir7, city)
            target_dir = os.path.join(self.targets_dir7, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir8):
            img_dir = os.path.join(self.images_dir8, city)
            target_dir = os.path.join(self.targets_dir8, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir9):
            img_dir = os.path.join(self.images_dir9, city)
            target_dir = os.path.join(self.targets_dir9, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir10):
            img_dir = os.path.join(self.images_dir10, city)
            target_dir = os.path.join(self.targets_dir10, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir11):
            img_dir = os.path.join(self.images_dir11, city)
            target_dir = os.path.join(self.targets_dir11, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir12):
            img_dir = os.path.join(self.images_dir12, city)
            target_dir = os.path.join(self.targets_dir12, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir13):
            img_dir = os.path.join(self.images_dir13, city)
            target_dir = os.path.join(self.targets_dir13, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir14):
            img_dir = os.path.join(self.images_dir14, city)
            target_dir = os.path.join(self.targets_dir14, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir15):
            img_dir = os.path.join(self.images_dir15, city)
            target_dir = os.path.join(self.targets_dir15, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

        for city in os.listdir(self.images_dir16):
            img_dir = os.path.join(self.images_dir16, city)
            target_dir = os.path.join(self.targets_dir16, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)

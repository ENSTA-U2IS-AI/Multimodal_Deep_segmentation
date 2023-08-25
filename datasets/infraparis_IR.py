import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class INFRAPARIS_IR(data.Dataset):
    """INFRAPARIS dataset Dataset, close from the Cityscapes Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'Left' and 'semantic_segmentation_truth' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="semantic_segmentation_truth" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'semantic_segmentation_truth' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    INFRAPARISClass = namedtuple('CityscapesClass', ['name', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        INFRAPARISClass('road',                 0, 'flat', 1, False, False, (224, 92, 94)),
        INFRAPARISClass('sidewalk',             1, 'flat', 1, False, False, (98, 229, 212)),
        INFRAPARISClass('building',             2, 'construction', 2, False, False, (75, 213, 234)),
        INFRAPARISClass('wall',                 3, 'construction', 2, False, False, (42, 186, 83)),
        INFRAPARISClass('fence',                4, 'construction', 2, False, False, (65, 255, 12)),
        INFRAPARISClass('pole',                 5, 'object', 3, False, False, (46, 181, 211)),
        INFRAPARISClass('traffic light',        6, 'object', 3, False, False, (38, 173, 42)),
        INFRAPARISClass('traffic sign',         7, 'object', 3, False, False, (237, 61, 222)),
        INFRAPARISClass('vegetation',           8, 'nature', 4, False, False, (122, 234, 2)),
        INFRAPARISClass('terrain',              9, 'nature', 4, False, False, (86, 244, 247)),
        INFRAPARISClass('sky',                  10, 'sky', 5, False, False, (87, 242, 87)),
        INFRAPARISClass('person',               11, 'human', 6, True, False, (33, 188, 119)),
        INFRAPARISClass('rider',                12, 'human', 6, True, False, (216, 36, 186)),
        INFRAPARISClass('car',                  13, 'vehicle', 7, True, False, (224, 172, 51)),
        INFRAPARISClass('truck',                14, 'vehicle', 7, True, False, (232, 196, 97)),
        INFRAPARISClass('bus',                  15, 'vehicle', 7, True, False, (0, 137, 150)),
        INFRAPARISClass('train',                16, 'vehicle', 7, True, False, (97, 232, 187)),
        INFRAPARISClass('motorcycle',           17, 'vehicle', 7, True, False, (239, 107, 197)),
        INFRAPARISClass('bicycle',              18, 'vehicle', 7, True, False, (149, 15, 252)),
        INFRAPARISClass('unlabeled',            255, 'void', 0, False, True, (206, 140, 26)),
    ]

    train_id_to_color = [c.color for c in classes ]
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'semantic_segmentation_reprojected'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'Infra', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = file_name.replace('.png','_labelIds.png')
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
        image = Image.open(self.images[index])#.convert('RGB')
        image = np.asarray(image)
        h,w= np.shape(image)
        image = 255*image.astype(np.float)/np.max(image)
        image_IR=np.zeros((h,w,3))
        image_IR[:, :, 0] = image
        image_IR[:, :, 1] = image
        image_IR[:, :, 2] = image
        image_IR = Image.fromarray(image_IR.astype(np.uint8))


        target = Image.open(self.targets[index])
        #print('image.size()', np.shape(target))
        if self.transform:
            image_IR, target = self.transform(image_IR, target)
        target = self.encode_target(target)
        return image_IR, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

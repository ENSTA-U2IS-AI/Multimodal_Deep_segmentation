import json
import os
from collections import namedtuple

import torch.utils.data as data
import numpy as np
import cv2 as cv
from PIL import Image

class dataset(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    datasetClass = namedtuple('datasetClass', ['name', 'id', 'train_id', 'object_id', 'instance', 'road_layer'])
    classes = [
        datasetClass('bots',                                        0, 255, (50, 0, 150), False, False),
        datasetClass('traffic_lights_head',                         1, 0, (0, 0, 255), True, False),
        datasetClass('mailbox',                                     2, 255, (200, 50, 50), False, False),
        datasetClass('tram_tracks',                                 3, 255, (130, 90, 255), False, False),
        datasetClass('cyclist',                                     4, 255, (250, 50, 255), True, False),
        datasetClass('traffic_lights_bulb_yellow',                  5, 255, (0, 0, 253), False, False),
        datasetClass('barbacue',                                    6, 255, (200, 60, 200), False, False),
        datasetClass('vase',                                        7, 255, (145, 190, 25), False, False),
        datasetClass('poster',                                      8, 255, (90, 0, 90), False, False),
        datasetClass('yellow_barrel',                               9, 255, (255, 0, 50), False, False),
        datasetClass('vegetation',                                  10, 1, (0, 255, 255), False, False),
        datasetClass('motorcycle',                                  11, 255, (50, 100, 100), True, False),
        datasetClass('house',                                       12, 255, (130, 0, 0), False, False),
        datasetClass('construction_helmet_green',                   13, 255, (0, 80, 205), True, False),
        datasetClass('electric_post_insulator_break',               14, 255, (100, 0, 150), False, False),
        datasetClass('uninteresting',                               15, 255, (1, 1, 1), False, False),
        datasetClass('wood',                                        16, 255, (201, 100, 0), False, False),
        datasetClass('construction_scaffold',                       17, 255, (112, 132, 112), False, False),
        datasetClass('car',                                         18, 2, (100, 75, 0), True, False),
        datasetClass('hammock',                                     19, 255, (50, 125, 100), False, False),
        datasetClass('construction_cord',                           20, 255, (22, 82, 152), False, False),
        datasetClass('rock',                                        21, 255, (255, 255, 150), False, False),
        datasetClass('suitcase',                                    22, 255, (210, 135, 246), True, False),
        datasetClass('construction_fence',                          23, 255, (50, 80, 50), False, False),
        datasetClass('garbage_bag',                                 24, 255, (130, 130, 130), False, False),
        datasetClass('bicycle',                                     25, 255, (150, 10, 60), True, False),
        datasetClass('tree_pit',                                    26, 3, (0, 255, 100), False, False),
        datasetClass('electric_power',                              27, 255, (50, 110, 10), False, False),
        datasetClass('construction_helmet_white',                   28, 255, (0, 80, 201), True, False),
        datasetClass('wind_direction',                              29, 255, (100, 10, 100), False, False),
        datasetClass('safety_vest_03_pink',                         30, 255, (0, 81, 103), True, False),
        datasetClass('segway',                                      31, 255, (20, 80, 50), False, False),
        datasetClass('lane_bike',                                   32, 255, (200, 150, 250), False, False),
        datasetClass('crosswalk',                                   33, 4, (150, 255, 255), False, True),
        datasetClass('baby_cart',                                   34, 255, (210, 130, 240), True, False),
        datasetClass('pile_of_sand',                                35, 255, (35, 100, 40), False, False),
        datasetClass('traffic_signs_poles_or_structure',            36, 255, (255, 150, 0), False, False),
        datasetClass('safety_vest_03_yellow',                       37, 255, (0, 80, 103), True, False),
        datasetClass('parking_area',                                38, 255, (0, 50, 255), False, False),
        datasetClass('dog',                                         39, 255, (200, 150, 255), True, False),
        datasetClass('wheel_chair',                                 40, 255, (210, 135, 247), True, False),
        datasetClass('safety_vest_03_green',                        41, 255, (0, 85, 103), True, False),
        datasetClass('garbage_road',                                42, 255, (50, 50, 51), False, True),
        datasetClass('concrete_benchs',                             43, 255, (150, 10, 110), False, False),
        datasetClass('umbrella_garden',                             44, 255, (200, 180, 20), False, False),
        datasetClass('bus',                                         45, 255, (100, 100, 150), True, False),
        datasetClass('safety_vest_04',                              46, 255, (0, 80, 104), True, False),
        datasetClass('pool',                                        47, 255, (140, 200, 170), False, False),
        datasetClass('building',                                    48, 5, (150, 0, 255), False, False),
        datasetClass('truck',                                       49, 255, (50, 100, 200), True, False),
        datasetClass('road_lines',                                  50, 6, (50, 200, 10), False, True),
        datasetClass('lamp',                                        51, 255, (30, 190, 100), True, False),
        datasetClass('water',                                       52, 255, (130, 160, 210), False, False),
        datasetClass('wall',                                        53, 255, (230, 230, 230), False, False),
        datasetClass('portable_bathroom',                           54, 255, (155, 155, 250), False, False),
        datasetClass('construction_post_cone',                      55, 255, (110, 130, 110), False, False),
        datasetClass('armchair',                                    56, 255, (20, 180, 44), False, False),
        datasetClass('ego_car',                                     57, 255, (5, 5, 5), False, False),
        datasetClass('kerb_stone',                                  58, 7, (20, 40, 80), False, False),
        datasetClass('air_conditioning',                            59, 255, (98, 110, 10), False, False),
        datasetClass('safety_vest_03_red',                          60, 255, (0, 83, 103), True, False),
        datasetClass('press_box',                                   61, 255, (90, 150, 230), False, False),
        datasetClass('biker',                                       62, 255, (250, 100, 255), True, False),
        datasetClass('table',                                       63, 255, (10, 255, 140), False, False),
        datasetClass('tv_antenna',                                  64, 255, (100, 10, 200), False, False),
        datasetClass('beacon_light',                                65, 255, (200, 10, 100), False, False),
        datasetClass('parking_bicycles',                            66, 255, (0, 0, 100), False, False),
        datasetClass('longitudinal_crack',                          67, 255, (213, 128, 0), True, True),
        datasetClass('vending_machine',                             68, 255, (31, 31, 181), False, False),
        datasetClass('tricycle',                                    69, 255, (216, 0, 89), False, False),
        datasetClass('walker',                                      70, 255, (210, 135, 245), True, False),
        datasetClass('chair',                                       71, 255, (5, 95, 10), False, False),
        datasetClass('safety_vest_03_turquoise',                    72, 255, (0, 86, 103), True, False),
        datasetClass('swing',                                       73, 255, (130, 70, 240), False, False),
        datasetClass('electric_post_conductor',                     74, 255, (116, 116, 116), False, False),
        datasetClass('construction_helmet_blue',                    75, 255, (0, 80, 204), True, False),
        datasetClass('umbrella',                                    76, 255, (210, 135, 248), True, False),
        datasetClass('ball',                                        77, 255, (210, 135, 249), True, False),
        datasetClass('sidewalk',                                    78, 8, (150, 200, 130), False, False),
        datasetClass('construction_helmet_red',                     79, 255, (0, 80, 203), True, False),
        datasetClass('jersey_barrier',                              80, 255, (255, 150, 255), False, False),
        datasetClass('traffic_signs',                               81, 255, (255, 0, 255), False, False),
        datasetClass('terrace',                                     82, 255, (2, 10, 13), False, False),
        datasetClass('container',                                   83, 255, (255, 255, 200), False, False),
        datasetClass('transversal_crack',                           84, 255, (213, 213, 0), True, False),
        datasetClass('kerb_rising_edge',                            85, 9, (20, 40, 90), False, False),
        datasetClass('construction_contrainer',                     86, 10, (113, 133, 113), False, False),
        datasetClass('construction_concrete',                       87, 255, (20, 80, 150), False, False),
        datasetClass('train',                                       88, 255, (30, 80, 230), True, False),
        datasetClass('dog_house',                                   89, 255, (50, 110, 240), False, False),
        datasetClass('water_tank',                                  90, 255, (255, 100, 50), False, False),
        datasetClass('painting',                                    91, 255, (45, 190, 240), False, False),
        datasetClass('playground',                                  92, 255, (80, 80, 80), False, False),
        datasetClass('bench',                                       93, 255, (0, 130, 150), False, False),
        datasetClass('safety_vest_03_orange',                       94, 255, (0, 84, 103), True, False),
        datasetClass('plumbing',                                    95, 255, (190, 150, 230), False, False),
        datasetClass('pergola_garden',                              96, 255, (150, 200, 50), False, False),
        datasetClass('vegetation_road',                             97, 255, (0, 255, 254), False, True),
        datasetClass('stairs',                                      98, 255, (90, 10, 90), False, False),
        datasetClass('safety_vest_02',                              99, 255, (0, 80, 102), True, False),
        datasetClass('sunshades',                                   100, 255, (255, 50, 0), False, False),
        datasetClass('van',                                         101, 11, (185, 255, 75), True, False),
        datasetClass('railings',                                    102, 255, (100, 50, 0), False, False),
        datasetClass('kickbike',                                    103, 255, (185, 255, 46), True, False),
        datasetClass('bin',                                         104, 255, (0, 50, 0), False, False),
        datasetClass('scooter_child',                               105, 255, (225, 35, 25), False, False),
        datasetClass('construction_stock',                          106, 255, (25, 85, 155), False, False),
        datasetClass('traffic_cameras',                             107, 255, (50, 150, 255), False, False),
        datasetClass('construction_pallet',                         108, 255, (111, 131, 111), False, False),
        datasetClass('road',                                        109, 12, (150, 150, 200), False, False),
        datasetClass('electric_post_insulator',                     110, 255, (100, 0, 100), False, False),
        datasetClass('traffic_signs_back',                          111, 255, (0, 200, 0), False, False),
        datasetClass('decoration_garden',                           112, 255, (250, 100, 130), False, False),
        datasetClass('garbage',                                     113, 255, (50, 50, 50), False, False),
        datasetClass('traffic_lights_bulb_green',                   114, 255, (0, 0, 252), False, False),
        datasetClass('cat_ete',                                     115, 255, (5, 50, 150), False, True),
        datasetClass('marquees',                                    116, 255, (180, 90, 30), False, False),
        datasetClass('asphalt_hole',                                117, 255, (213, 128, 213), False, True),
        datasetClass('alley',                                       118, 255, (10, 80, 15), False, False),
        datasetClass('bridge',                                      119, 255, (30, 90, 210), False, False),
        datasetClass('hangar_airport',                              120, 255, (150, 0, 150), False, False),
        datasetClass('billboard',                                   121, 255, (255, 100, 0), False, False),
        datasetClass('barrel',                                      122, 255, (205, 255, 205), False, False),
        datasetClass('toy',                                         123, 255, (60, 30, 255), False, False),
        datasetClass('subway',                                      124, 255, (200, 225, 90), False, False),
        datasetClass('plane',                                       125, 255, (35, 65, 50), False, False),
        datasetClass('food_machine',                                126, 255, (255, 128, 0), False, False),
        datasetClass('runway',                                      127, 255, (50, 50, 200), False, False),
        datasetClass('fire',                                        128, 255, (255, 0, 1), False, False),
        datasetClass('phone_booth',                                 129, 255, (30, 30, 180), False, False),
        datasetClass('skateboard',                                  130, 255, (185, 200, 185), True, False),
        datasetClass('box',                                         131, 255, (110, 90, 225), False, False),
        datasetClass('polished_aggregated',                         132, 255, (213, 0, 0), True, True),
        datasetClass('fire_extinguisher',                           133, 255, (100, 80, 100), False, False),
        datasetClass('people',                                      134, 13, (250, 150, 255), True, False),
        datasetClass('traffic_lights_bulb_red',                     135, 255, (0, 0, 254), False, False),
        datasetClass('sewer_road',                                  136, 255, (255, 0, 1), False, True),
        datasetClass('traffic_lights_poles',                        137, 14, (0, 255, 0), False, False),
        datasetClass('fire_hydrant',                                138, 255, (255, 255, 0), False, False),
        datasetClass('terrain',                                     139, 255, (150, 100, 0), False, False),
        datasetClass('tools',                                       140, 255, (160, 25, 230), False, False),
        datasetClass('safety_vest_01',                              141, 255, (0, 80, 101), True, False),
        datasetClass('safety_vest_03_blue',                         142, 255, (0, 82, 103), True, False),
        datasetClass('sewer',                                       143, 15, (255, 0, 0), False, False),
        datasetClass('stony_floor',                                 144, 255, (150, 100, 100), False, False),
        datasetClass('electric_post',                               145, 255, (100, 0, 200), False, False),
        datasetClass('construction_helmet_yellow',                  146, 255, (0, 80, 200), True, False),
        datasetClass('carpet',                                      147, 255, (230, 30, 130), False, False),
        datasetClass('sky',                                         148, 16, (0, 0, 0), False, False),
        datasetClass('trash_can',                                   149, 255, (255, 150, 150), False, False),
        datasetClass('pivot',                                       150, 255, (135, 75, 180), False, False),
        datasetClass('fences',                                      151, 17, (150, 200, 250), False, False),
        datasetClass('construction_helment_orange',                 152, 255, (0, 80, 202), True, False),
        datasetClass('street_lights',                               153, 18, (0, 175, 100), False, False),
        datasetClass('bird',                                        154, 255, (255, 175, 110), False, False),
    ]

    train_id_to_color = [c.object_id for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root_dataset, root_odgt, split, transform=None):
        self.root_dataset = root_dataset
        self.transform = transform
        self.list_sample = [json.loads(x.rstrip()) for x in open(root_odgt + '/' + split + '.odgt', 'r')]

    # @classmethod
    # def encode_target(cls, target):
    #     return cls.id_to_train_id[np.array(target)]
    #
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
        this_record = self.list_sample[index]
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        segm = cv.imread(segm_path)
        #print('segm_path',segm_path,np.shape(segm))
        segm = cv.cvtColor(segm, cv.COLOR_BGR2RGB)
        target = np.zeros((segm.shape[0], segm.shape[1])) + 255

        for c in self.classes:
            upper = np.array(c.object_id)
            lower = upper
            mask = cv.inRange(segm, lower, upper)
            target[mask == 255] = c.train_id
        target = target.astype(np.uint8)
        #target = cv.cvtColor(target, cv.COLOR_GRAY2BGR)
        target = Image.fromarray(target)
        image = Image.fromarray(image)
        if self.transform:
            image, target = self.transform(image, target)
        # target = self.encode_target(target)
    

        return image, target

    def __len__(self):
        return len(self.list_sample)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data


if __name__ == '__main__':

    train_data = dataset(root_dataset='../av-challenge', root_odgt='./data', split='val')
    train_loader = data.DataLoader(
        train_data, batch_size=1, shuffle=True, num_workers=2)
    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=1, pin_memory=True, num_workers=16, shuffle=True, drop_last=True)
    for (images, labels) in train_loader:
        print(f'{images.shape}', f'{labels.shape}')

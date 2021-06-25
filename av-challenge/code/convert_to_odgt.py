import json
from pathlib import Path
import os
import cv2 as cv

def convert_to_odgt(odgt, image_path, label_path):
    with odgt.open(mode='w+') as fo:
        od_line = {}
        file_name = os.listdir(image_path)
        for file in file_name:
            od_line['fpath_img'] = image_path.split('/')[-3] + '/' + 'leftImg8bit' + '/' + file
            label = file.split('leftImg8bit')[0] + 'leftLabel.png'
            od_line['fpath_segm'] = label_path.split('/')[-3] + '/' + 'leftLabel' + '/' + label
            img = cv.imread(image_path + file)
            od_line['width'] = img.shape[1]
            od_line['height'] = img.shape[0]
            fo.write(f'{json.dumps(od_line)}\n')

if __name__ == "__main__":

    train_odgt = Path('../data_odgt/train.odgt')
    val_odgt = Path('../data_odgt/val.odgt')
    train_path = '../train/leftImg8bit/'
    train_label_path = '../train/leftLabel/'
    val_path = '../val/leftImg8bit/'
    val_label_path = '../val/leftLabel/'

    convert_to_odgt(train_odgt, train_path, train_label_path)
    convert_to_odgt(val_odgt, val_path, val_label_path)

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

    test1_odgt = Path('../data_odgt/test_normal.odgt')
    test2_odgt = Path('../data_odgt/test_OOD.odgt')
    test3_odgt = Path('../data_odgt/test_level1.odgt')
    test4_odgt = Path('../data_odgt/test_level2.odgt')

    test1_path = '/home/soumik/workspace_hard1/workspacegianni/BDDs/SORTED_dataset_synthetic/test_normal/leftImg8bit/'
    test1_label_path = '/home/soumik/workspace_hard1/workspacegianni/BDDs/SORTED_dataset_synthetic/test_normal/leftLabel/'

    test2_path = '/home/soumik/workspace_hard1/workspacegianni/BDDs/SORTED_dataset_synthetic/test_OOD/leftImg8bit/'
    test2_label_path = '/home/soumik/workspace_hard1/workspacegianni/BDDs/SORTED_dataset_synthetic/test_OOD/leftLabel/'

    test3_path = '/home/soumik/workspace_hard1/workspacegianni/BDDs/SORTED_dataset_synthetic/test_level1/leftImg8bit/'
    test3_label_path = '/home/soumik/workspace_hard1/workspacegianni/BDDs/SORTED_dataset_synthetic/test_level1/leftLabel/'

    test4_path = '/home/soumik/workspace_hard1/workspacegianni/BDDs/SORTED_dataset_synthetic/test_level2/leftImg8bit/'
    test4_label_path = '/home/soumik/workspace_hard1/workspacegianni/BDDs/SORTED_dataset_synthetic/test_level2/leftLabel/'

    convert_to_odgt(test1_odgt, test1_path, test1_label_path)
    convert_to_odgt(test2_odgt, test2_path, test2_label_path)
    convert_to_odgt(test3_odgt, test3_path, test3_label_path)
    convert_to_odgt(test4_odgt, test4_path, test4_label_path)


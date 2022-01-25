import h5py

import torch
import torch.utils.data
import numpy as np
import cv2 as cv

class KITTI_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path='', mode='training'):

        self.dataset_path = dataset_path
        self.mode = mode

    def __getitem__(self, idx):

        info_file = h5py.File(self.dataset_path, 'r') #파일 열기

        idx=str(idx).zfill(10)

        info_location = '/' + self.mode + '_group/' + idx
        info = list(info_file[info_location])
        for i in range(7):
            info[i] = str(info[i], 'utf-8')

        image = cv.imread(info[0])

        pose_list = [info[1], info[2], info[3], info[4], info[5], info[6]]

        info_file.close()

        return image, pose_list

    def __len__(self):

        file = h5py.File(self.dataset_path, 'r') #파일 열기
        self.len = file.get('/' + self.mode + '_group/').__len__()
        file.close()

        return self.len

#mode : training / validation / test
if __name__ == '__main__':
    kitti_dataset = KITTI_dataset(dataset_path='KITTI_dataset_info.hdf5', mode='training')
    kitti_dataset.__getitem__(0)
    kitti_dataset.__len__()

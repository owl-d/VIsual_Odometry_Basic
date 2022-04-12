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
        for i in range(14):
            info[i] = str(info[i], 'utf-8')
        for i in range(12):
            info[i+2] = float(info[i+2])

        prev_image = cv.imread(info[0])
        current_image = cv.imread(info[1])
        stack_image = np.concatenate((current_image, prev_image), axis=2) # 현재 이미지가 위(1~3채널), 과거 이미지가 아래(4~6채널)인 6채널 이미지
        
        prev_pose_list = [info[2], info[3], info[4], info[5], info[6], info[7]]
        current_pose_list = [info[8], info[9], info[10], info[11], info[12], info[13]]
        diff_pose_list = []

        for i in range(6):
            diff_pose_list.append(current_pose_list[i] - prev_pose_list[i])  # T - (T-1) : 두 시간 DoF의 차이값
        diff_pose_list = torch.Tensor(diff_pose_list)

        info_file.close()

        return stack_image, diff_pose_list

    def __len__(self):

        file = h5py.File(self.dataset_path, 'r') #파일 열기
        self.len = file.get('/' + self.mode + '_group/').__len__()
        file.close()

        return self.len

#mode : training / validation / test
if __name__ == '__main__':
    kitti_dataset = KITTI_dataset(dataset_path='KITTI_stack_dataset_info.hdf5', mode='training')
    kitti_dataset.__getitem__(0)
    kitti_dataset.__len__()

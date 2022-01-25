from torch.utils.data import DataLoader

import cv2 as cv
import numpy as np

from dataset_generator_KITTI import KITTI_dataset_generator
from dataloader_KITTI import KITTI_dataset



file_make = False

if file_make == True : #True이면 h5py 파일 생성
    kitti_dataset_generator = KITTI_dataset_generator(dataset_save_path='KITTI_dataset_info.hdf5',
                                dataset_path='/media/doyu/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color')

#데이터셋
kitti_dataset_training = KITTI_dataset(dataset_path='KITTI_dataset_info.hdf5', mode='training')
kitti_dataset_validation = KITTI_dataset(dataset_path='KITTI_dataset_info.hdf5', mode='validation')
kitti_dataset_test = KITTI_dataset(dataset_path='KITTI_dataset_info.hdf5', mode='test')

#데이터로더
training_dataloader = DataLoader(dataset=kitti_dataset_training, batch_size=1, shuffle=False, num_workers=0)
validation_dataloader = DataLoader(dataset=kitti_dataset_validation, batch_size=1, shuffle=False,num_workers=0)
test_dataloader = DataLoader(dataset=kitti_dataset_test, batch_size=1, shuffle=False, num_workers=0)


# if __name__ == '__main__':
for i_batch, (image, pose_list) in enumerate(training_dataloader):

    # # 6DoF 출력
    print('batch', i_batch, end=' ')
    print("pose(6DoF) : \n x : {}, y : {}, z : {} \n roll : {}, pitch : {}, yaw : {}"
                .format(pose_list[0], pose_list[1], pose_list[2], pose_list[3], pose_list[4], pose_list[5]))

    # # 해당 이미지 출력
    image = image.numpy()
    image = image.reshape(376, 1241, 3)
    cv.imshow('img', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 처음 두 개 샘플만 보여주기
    if i_batch == 1:
        break

from torch.utils.data import DataLoader

import cv2 as cv

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
training_dataloader = DataLoader(dataset=kitti_dataset_training, batch_size=1, shuffle=True, num_workers=0)
validation_dataloader = DataLoader(dataset=kitti_dataset_validation, batch_size=1, shuffle=True,num_workers=0)
test_dataloader = DataLoader(dataset=kitti_dataset_test, batch_size=1, shuffle=True, num_workers=0)


# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(training_dataloader):

    print(sample_batched)
    # 6DoF 출력
    print("pose(6DoF) : \n x : {}, y : {}, z : {} \n roll : {}, pitch : {}, yaw : {}"
                .format(sample_batched['x'], sample_batched['y'], sample_batched['z'],
                sample_batched['roll'], sample_batched['pitch'], sample_batched['yaw']))

    # 해당 이미지 출력
    idx_img_path = sample_batched['path']
    print(idx_img_path)
    # idx_img = cv.imread(idx_img_path, cv.IMREAD_COLOR)
    # cv.imshow('idx_img', idx_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 처음 두 개 샘플만 보여주기
    if i_batch == 1:
        break
import h5py
import cv2 as cv


class KITTI_dataset():

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

        # 6DoF 출력
        print("pose(6DoF) : \n x : {}, y : {}, z : {} \n roll : {}, pitch : {}, yaw : {}"
        .format(info[1], info[2], info[3], info[4], info[5], info[6]))

        # 해당 이미지 출력
        idx_img_path = info[0]
        idx_img = cv.imread(idx_img_path, cv.IMREAD_COLOR)
        cv.imshow('idx_img', idx_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        info_file.close()

#mode : training / validation / test
kitti_data = KITTI_dataset(dataset_path='KITTI_dataset_info.hdf5', mode='training')
kitti_data.__getitem__(0)

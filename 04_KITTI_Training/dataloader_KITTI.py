import h5py

import torch
import torch.utils.data

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

        sample = {'path': info[0], 'x': info[1], 'y' : info[2], 'z' : info[3],
                    'roll' : info[4], 'pitch' : info[5], 'yaw' : info[6]}

        info_file.close()

        return sample

    def __len__(self):

        file = h5py.File(self.dataset_path, 'r') #파일 열기
        self.len = file.get('/' + self.mode + '_group/').__len__()
        file.close()

        return self.len

#mode : training / validation / test
if __name__ == '__main__':
    kitti_dataset = KITTI_dataset(dataset_path='KITTI_dataset_info.hdf5', mode='training')
    kitti_dataset.__getitem__(10)
    kitti_dataset.__len__()

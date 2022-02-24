import h5py

import os

import numpy as np
import math


class KITTI_dataset_generator():

    def __init__(self,  dataset_save_path='',
                        dataset_path=''):
        
        ### Seauence Division ######################################################
        self.train_sequence = ["00", "01", "03", "04", "10"]
        self.valid_sequence = ["02", "05", "06"]
        self.test_sequence = ["07", "08", "09"]
        ############################################################################

        self.dataset_save_path = dataset_save_path
        
        self.image_dataset_path = dataset_path + '/dataset/sequences'
        self.pose_dataset_path = dataset_path + '/data_odometry_poses/dataset/poses'


        ### Dataset HDF Preparation by Dataset Group Type ###########################
        main_file = h5py.File(self.dataset_save_path, 'w')

        self.train_group = main_file.create_group('training_group')
        self.train_group.attrs['type'] = 'training'

        self.valid_group = main_file.create_group('validation_group')
        self.valid_group.attrs['type'] = 'validation'

        self.test_group = main_file.create_group('test_group')
        self.test_group.attrs['type'] = 'test'
        #############################################################################

        self.getpose(self.train_sequence, self.train_group)
        self.getpose(self.valid_sequence, self.valid_group)
        self.getpose(self.test_sequence, self.test_group)
        main_file.close()

    def getpose(self, group_sequence, sub_group):

        idx = 0
        for sequence in group_sequence:

            seq_pose_file = open(self.pose_dataset_path + '/' + sequence + '.txt', 'r') #pose.txt 파일 열기
            seq_img_list = sorted(os.listdir(self.image_dataset_path + '/' + sequence + '/image_2')) #해당 시컨스의 이미지 파일 정렬 리스트

            for seq_idx in seq_img_list:
                
                ### transformation matix 로 pose 계산 : 6DoF #############################################################################
                matrix = seq_pose_file.readline() #transformation matrix
                pose = matrix.strip().split()
                pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])  #translation matrix
                pose_R = np.array([[float(pose[0]), float(pose[1]), float(pose[2])],  #rotation matrix
                                    [float(pose[4]), float(pose[5]), float(pose[6])], 
                                    [float(pose[8]), float(pose[9]), float(pose[10])]])

                x = pose_T[0] #position
                y = pose_T[1]
                z = pose_T[2]
                roll = math.atan2(pose_R[2][1], pose_R[2][2]) #orientation
                pitch = math.atan2(-pose_R[2][0], math.sqrt(pose_R[2][1]**2 + pose_R[2][2]**2))
                yaw = math.atan2(pose_R[1][0], pose_R[0][0])
                ########################################################################################################################
                
                # 샘플 별 이미지 path, pose h5py에 입력
                img_path=str(self.image_dataset_path + '/' + sequence + '/image_2' + '/' + seq_idx).encode('utf-8') #utf-8로 encoding
                                                                                                                    #image_2 : 두번째 컬러 카메라
                sub_group.create_dataset(name=str(idx).zfill(10),
                                            data=[img_path, x, y, z, roll, pitch, yaw],
                                            compression='gzip', compression_opts=9)     #90% 압축하여 저장 속도를 빠르게 할 수 있다.
                                                                                        #만약 이미지 경로가 아닌 이미지 자체를 넣는 등 파일이 커지는 경우 효과적이다.
                
                idx += 1

            seq_pose_file.close()


if __name__ == '__main__': 
    kitti = KITTI_dataset_generator(dataset_save_path='KITTI_dataset_info.hdf5',
                                dataset_path='/media/doyu/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color')


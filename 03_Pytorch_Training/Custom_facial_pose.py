from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Warnings 무시하기
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


###### Example(index 65) ################################################################################
landmarks_frame = pd.read_csv('./03_Pytorch_Training/data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]       #이미지 이름
landmarks = landmarks_frame.iloc[n, 1:]     #n번째 행 접근
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)    #(해당 row의 landmark 수, 2)로 reshape

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4])) #처음 4개 landmark


### sample 보여주기 위해 이미지와 랜드마크 보여주는 함수 ####################################
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
####################################################################################

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
plt.show()
#########################################################################################################


# 1. Dataset class

class FaceLandmarksDataset(Dataset): #Face Landmarks dataset

    def __init__(self, csv_file, root_dir, transform=None):     #csv_file(String) : csv file 경로
                                                                #root_dir(String) : 모든 이미지 포함한 디렉토리
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):  #데이터셋 길이(=학습해야 하는 길이) 반환
        return len(self.landmarks_frame)

    def __getitem__(self, idx): #index 'idx'에 따라 이미지와 랜드마크 읽어온다. 필요에 따라 읽어오므로 메모리 절약 가능
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


face_dataset = FaceLandmarksDataset(csv_file='./03_Pytorch_Training/data/faces/face_landmarks.csv',
                                    root_dir='./03_Pytorch_Training/data/faces')


#처음 4개 샘플의 크기와 랜드마크 보여준다.
fig = plt.figure()
for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


# 2. Transform
# 샘플의 사이즈가 정해져 있지 않다. 이를 고정된 크기로 바꿔준다.

class Rescale(object):  #이미지를 고정된 사이즈로 변경한다.
                        #매개변수는 tuple이거나 int 형식이다.
                        #tuple이면 이에 따라 고정된 사이즈로 출력하고, int이면 비율 유지하면서 더 짧은 가장자리의 길이와 맞춘다.
                                                    

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):   #샘플의 이미지를 랜덤하게 자른다.
                            #매개변수는 tuple이거나 int 형식이다.
                            #tuple이면 이에 따라 고정된 사이즈로 출력하고, int이면 square crop이다.

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object): # 샘플의 ndarrays를 텐서로 변환

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # column 순서 바꿔준다
        # numpy 이미지: H x W x C (channel last)
        # torch 이미지: C x H x W (channel first)
        image = image.transpose((2, 0, 1))  #original shape으로부터 0차원->1차원 / 1차원->2차원 / 2차원->0차원
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


# 3. Transform 적용
scale = Rescale(256)  #이미지의 짧은 가장자리 길이가 256이 되도록 Rescale
crop = RandomCrop(128) #128 크기로 squre crop
composed = transforms.Compose([Rescale(256), RandomCrop(224)]) #이미지의 짧은 가장자리 길이가 256이 되도록 Rescale, 224 크기로 squre crop

# 샘플에 위 transform 적용
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    #적용된 예시 보여주기
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()




# 4. Iterating through the dataset
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

# 샘플 4개 출력해보기
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(),        #[3, 224, 224]
             sample['landmarks'].size())    #[68, 2]

    if i == 3:
        break


dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)
#DataLoader 파라미터 :    #dataset : 로드해 올 데이터의 데이터셋
                        #batch_size : 한 배치당 로드해 올 샘플의 수를 정한다.
                        #shuffle : True이면 데어터의 순서를 섞어 오버피팅(과대적합)을 예방한다.
                        #sampler : shuffle=False일 때 데이터의 인덱스를 조정할 수 있다.
                        #batch_sampler : sampler와 비슷하지만, 한 번에 인덱스의 배치 반환한다.
                        #num_workers : 데이터 로딩에 몇 개의 subprocess 돌리는지 정한다.
                        #collate_fn : 데이터셋에서 샘플 list를 batch 단위로 바꾸는 데 사용한다.
                        #pin_memory : cuda에게 일정 부분 메모리를 고정적으로 사용하겠다고 선언. 하지 않으면 RAM에서 GPU로 옮길 때 매번 다른 위치에 저장될 수 있다.
                        #drop_last : True일 때, batch size와 총 데이터 수가 나누어 떨어지지 않으면 버린다.
                        #timeout : 양수이면, 데이터로더가 데이터를 불러오는 제한시간이다.
                        #worker_init_fn : 어떤 worker를 불러올지 리스트 전달
                        #generator : RandomSampler가 랜덤 인덱스 생성하는 데 사용
                        #prefetch_factor : 학습 시작 전 일부 데이터를 미리 불러놓아 데이터 로딩 time을 줄인다. prefetch_factor*num_worker만큼 데이터 불러온다.
                        #persistent_workers : 에폭이 다 돌면 프로세스가 끝나는 게 일반적이지만, True로 해두면 여전히 살아 있어 프로세스 재생성하는 수고를 덜 수 있다.



def show_landmarks_batch(sample_batched): # 샘플의 batch를 랜드마크와 함께 보여주는 함수
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):   #i_batch=배치번호 , idx=전체 데이터에서 실제 인덱스 번호
    print(i_batch, sample_batched['image'].size(),  #[4, 3, 224, 224]
          sample_batched['landmarks'].size())       #[4, 68, 2]

    # batch 4개 보면 멈춘다.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


# # 5. 이런 식으로 Custom 해서 쓰면 된다 ~~
# import torch
# from torchvision import transforms, datasets

# data_transform = transforms.Compose([
#         transforms.RandomSizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
# hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                            transform=data_transform)
# dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                              batch_size=4, shuffle=True,
#                                              num_workers=4)
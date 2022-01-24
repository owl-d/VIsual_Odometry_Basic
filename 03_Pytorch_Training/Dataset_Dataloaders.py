#!/usr/bin/env python
# -*- coding: utf-8 -*-

#1. Loading a Dataset
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST( #Training dataset
    root="data", #path data is stored
    train=True, #it is training dataset
    download=True, #download the data from internet if it's not available at root
    transform=ToTensor() #specify the feature and label transformation
)

test_data = datasets.FashionMNIST( #test dataset
    root="data",
    train=False, #it is test dataset
    download=True,
    transform=ToTensor()
)


#2. Iterating and Visualizing the Dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8)) #show example image using matplotlib
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


#3. Creating a Custom Dataset for your files
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None): #데이터셋 오브젝트 생성 시 한 번 실행
        self.img_labels = pd.read_csv(annotations_file) 
        self.img_dir = img_dir                          #FashionMNIST 이미지는 "img_dir"에 저장되고
                                                        #labels은 CSV file "annotation_file"에 저장
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): #데이터셋에 존재하는 샘플의 수 반환
        return len(self.img_labels)

    def __getitem__(self, idx): #주어진 인덱스 idx에 따라 데이터셋의 샘플 로드하고 반환
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) #이미지 저장 경로
        image = read_image(img_path)                                        #이미지 읽어오기(텐서로 변환)
        label = self.img_labels.iloc[idx, 1]                                #일치하는 label 찾기
        if self.transform:                                                  #가능하다면 transform 함수
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label                                                 #텐서 이미지와 해당하는 label 튜플로 반환


#4. Preparing data for training with DataLoaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True) #Dataset은 데이터셋의 feature를 찾고 하나의 샘플에 레이블 지정한다.
                                                                    #Dalatloader는 이 과정을 편하게 해준다.
                                                                    #미니배치 : 한 번에 64개씩 이미지 가져와서 학습 또는 테스트
                                                                    #shffle : overfitting 방지하기 위해 매 에폭마다 이미지의 순서 섞는다


#5. Iterate through the DataLoader

train_features, train_labels = next(iter(train_dataloader)) #iteration마다 batch_size(=64)의 train_feature와 train_label 반환
print(f"Feature batch shape: {train_features.size()}") #[batch_size X 1 X 28 X 28]
print(f"Labels batch shape: {train_labels.size()}") #[batch_size]

# Display image and label
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
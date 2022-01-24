# -*- coding: utf-8 -*-

#1. Load and normalize CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose( #torchvision 데이터셋의 출력은 범위가 0 ~ 1인 PIL 이미지이다. 이를 범위 -1 ~ 1의 정규화된 텐서로 transform한다.
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2) #num_worker : 동시에 돌리는 subprocess 수 설정

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  #10개의 클래스

#####Show some of the training images###################################################
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
#######################################################################################


#2. Define a Convolutional Neural Network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # Convolution Layer1
        self.pool = nn.MaxPool2d(2, 2)      # MaxPooling Layer
        self.conv2 = nn.Conv2d(6, 16, 5)    # Convolution Layer2
        self.fc1 = nn.Linear(16*5*5, 120)   # fc Layer1 : 400 -> 120
        self.fc2 = nn.Linear(120, 84)       # fc Layer2 : 120 -> 84
        self.fc3 = nn.Linear(84, 10)        # fc Layer3 : 84  -> 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # Convolution1 : Conv2d(3,6,5)
                                                # in_channels=3 / out_channels=6 / kernel_size=5 / stride=1 / padding=0 / dilation=1
                                                # input : (N=4 , Cin=3 , Hin=32 , Win=32)
                                                # Output : (N=4 , Cout=6 , Hout=28 , Wout=28)

                                                # MaxPooling : Maxpool2d(2,2)
                                                # kernel_size=2 / stride=2 / padding=0 / dilation=1
                                                # input : (N=4 , Cin=6 , Hin=28 , Win=28)
                                                # Output : (N=4 , Cout=6 , Hout=14 , Wout=14)

        x = self.pool(F.relu(self.conv2(x)))    # Convolution2 : Conv2d(6, 16, 5)
                                                # in_channels=6 / out_channels=16 / kernel_size=5 / stride=1 / padding=0 / dilation=1
                                                # input : (N=4 , Cin=6 , Hin=14 , Win=14)
                                                # Output : (N=4 , Cout=16 , Hout=10 , Wout=10)

                                                # MaxPooling : Maxpool2d(2,2)
                                                # kernel_size=2 / stride=2 / padding=0 / dilation=1
                                                # input : (N=4 , Cin=16 , Hin=14 , Win=14)
                                                # Output : (N=4 , Cout=16 , Hout=5 , Wout=5)

        x = torch.flatten(x, 1)                 # Flatten : torch.flatten(x,1)
                                                # input = x (0차원 : batch_size , 1차원 : channel, 2차원 : height, 3차원 : width)
                                                # start_dim=1 : 1번 차원부터 모든 차원을 flatten 시킨다.
                                                # end_dim = -1
                                                # ouput = x (0차원 : batch_size=4 , 1차원 : channel*height*width = 16*5*5)
            
        x = F.relu(self.fc1(x))                 # fully_connected_1 : Linear(16*5*5, 120)
                                                # in_features=16*5*5 / out_features=120
                                                # input : (N=4 , *(해당없음) , Hin=16*5*5)
                                                # output : (N=4, *(해당없음) , Hout=120)

                                                # ReLU : functional.relu(x)
                                                # input = * (0차원 : 4 , 1차원 : 120)
                                                # output = * (0차원 : 4 , 1차원 : 120)

        x = F.relu(self.fc2(x))                 # fully_connected_2 : Linear(120, 84)
                                                # in_features=120 / out_features=84
                                                # input : (N=4 , *(해당없음) , Hin=120)
                                                # output : (N=4, *(해당없음) , Hout=84)

                                                # ReLU : functional.relu(x)
                                                # input = * (0차원 : 4 , 1차원 : 84)
                                                # output = * (0차원 : 4 , 1차원 : 84)

        x = self.fc3(x)                         # fully_connected_2 : Linear(84, 10)
                                                # in_features=84 / out_features=10
                                                # input : (N=4 , *(해당없음) , Hin=84)
                                                # output : (N=4, *(해당없음) , Hout=10)

        return x                                # x (0차원 : batch_size=4 , 1차원 : 10)

net = Net()

#3. Define a Loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()                               #Loss function으로 Cross-Entropy loss 사용
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #옵티마이저로 SGD 사용


#4. Train the network
for epoch in range(2):  # 모든 데이터셋을 도는 'epoch'을 여러번 반복한다.

    running_loss = 0.0  #새로운 에폭이 시작되었으므로 loss를 0으로 초기화한다. 
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data   #data 는 [inputs, labels] 형태의 리스트이다.

        # zero the parameter gradients
        optimizer.zero_grad() #gradients 관련 파라미터 초기화

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)   #Input=outputs(N, C) : net(input)의 추론한 결과이다.
                                            #Target=labels(N) : Groundtruth 번호로, N X 1 형태이다.
                                            #Output : reduction="None" 이면 텐서 벡터, "mean" 또는 "sum"이면 스칼라 형태이다. (Default=mean)
        loss.backward()                     
        optimizer.step()                    #옵티마이저 적용해 가중치 업데이트

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

#save trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)  #모델의 weight 만 저장


#5. Test the network on the test data
dataiter = iter(testloader)
images, labels = dataiter.next()

#print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


##############load back in saved model #####################################################
net = Net()
net.load_state_dict(torch.load(PATH)) #re-loading

output = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
###########################################################################################

#6. Accuracy
correct = 0
total = 0

with torch.no_grad():   #학습을 하지 않는 경우 gradient 계산할 필요가 없다.
    for data in testloader:     #테스트데이터에 대해
        images, labels = data   #데이터는 이미지와 레이블로 구성되어 있다.
        outputs = net(images)   #네트워크에 이미지를 넣어 계산해 output 얻는다.
        _, predicted = torch.max(outputs.data, 1)   #가장 큰 출력 가지는 클래스가 예측 클래스이다.
        total += labels.size(0)
        correct += (predicted == labels).sum().item() #예측 클래스와 실제 클래스가 동일하면 correct 수 증가

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %') #정확도는 (맞은 예측 수)/(전체 예측 수) 의 백분율

# 클래스 별 예측 결과
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():       # gradient 계산할 필요 없다.
    for data in testloader: # 테스트데이터셋에 대해
        images, labels = data
        outputs = net(images)   #이미지를 네트워크에 입력한다.
        _, predictions = torch.max(outputs, 1)  #가장 높은 클래스가 예측 클래스이다.

        # 클래스 별로 예측이 맞은 수 센다.
        for label, prediction in zip(labels, predictions):
            if label == prediction:                 #예측이 맞다면
                correct_pred[classes[label]] += 1   #해당 클래스의 정답 수 1 증가
            total_pred[classes[label]] += 1         #총 예측 횟수 1은 항상 증가


# 클래스 별 예측 결과 출력
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


#7. Training on GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #cuda machine 환경이면 "cuda:0", 아니면 "cpu"
print(device)

net.to(device) #cuda tensor로 변환 : GPU로 보낸다.

inputs, labels = data[0].to(device), data[1].to(device) #모든 단계의 input, label도 GPU로 보낸다.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


anno_train_df = pd.read_csv('./input/anno_train.csv', header=None)
anno_test_df = pd.read_csv('./input/anno_test.csv', header=None)
names_df = pd.read_csv('./input/names.csv', header=None)


car_data_dir = './input/car_data/'
train_dir = car_data_dir + 'train/'
test_dir = car_data_dir + 'test/'

# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, width, height, input_dir, anno_train_df, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for subdir, dirs, files in os.walk(input_dir):
            for f in files:
                if f == '.DS_Store': 
                    continue
                img_path = os.path.join(subdir, f)
                #img = Image.open(img_path)
                row = anno_train_df.loc[anno_train_df[0] == f]
                #box = (row[1], row[2], row[3], row[4])
                label = int(row[5])
                imgs.append((img_path,label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.width = width
        self.height = height

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        '''
        print(fn)
        print(label)
        '''
        img = self.loader(fn)
        img = img.resize((self.width, self.height),Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

train_transformations = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
test_transformations = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data=MyDataset(width = 1620, height = 1080, input_dir = train_dir,anno_train_df = anno_train_df, transform=train_transformations)
test_data=MyDataset(width = 1620, height = 1080,input_dir = test_dir,anno_train_df = anno_test_df, transform=test_transformations)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)


class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
 

        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleNet,self).__init__()
 
        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        #self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        #self.unit6 = Unit(in_channels=64, out_channels=64)
        #self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        #self.unit9 = Unit(in_channels=128, out_channels=128)
        #self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        #self.unit12 = Unit(in_channels=256, out_channels=256)
        #self.unit13 = Unit(in_channels=256, out_channels=256)
        #self.unit14 = Unit(in_channels=256, out_channels=256)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1, self.unit4, self.unit5, self.pool2, self.unit8, self.unit11, self.pool3, self.avgpool)
        #Add all the units into the Sequential layer in exact order
        '''
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)
        '''
        self.fc = nn.Linear(in_features=128,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,128)
        output = self.fc(output)
        return output


model = SimpleNet(num_classes = 196)
cuda_avail = torch.cuda.is_available()
if cuda_avail:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_func = torch.nn.CrossEntropyLoss()


def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_models(epoch):
    torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    print("Checkpoint saved")

def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):
      
        if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

        #Predict classes using images from the test set
        outputs = model(images)
        _,prediction = torch.max(outputs.data, 1)
        prediction = prediction.cpu().numpy()
        test_acc += torch.sum(prediction == labels.data)
        


    #Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 10000

    return test_acc

def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            #Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            #Clear all accumulated gradients
            optimizer.zero_grad()
            #Predict classes using images from the test set
            outputs = model(images)
            #Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs,labels)
            #Backpropagate the loss
            loss.backward()

            #Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data[0] * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            
            train_acc += torch.sum(prediction == labels.data)

        #Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        #Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 8041
        train_loss = train_loss / 8041

        #Evaluate on the test set
        test_acc = test()

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc


        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,test_acc))
train(50)
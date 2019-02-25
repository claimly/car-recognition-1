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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        '''
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        '''
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(128, 196)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        out = self.conv3(conv2_out)
        out = out.view(-1, self.num_flat_features(out))
        #conv4_out = self.conv4(conv3_out)
        out = self.dense(out)
        return out


    def num_flat_features(self, x):
        # 四维特征，第一维是batchSize
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = Net()
#model.cuda()
print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        #batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))
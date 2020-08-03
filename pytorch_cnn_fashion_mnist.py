import torch
import torch.nn as nn

from torch.autograd import Variable
import numpy as np
import pandas as pd
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils

raw_train_ds = pd.read_csv('./data/FashionMNIST/fashion-mnist_train.csv')
train_labels_ds = raw_train_ds.iloc[:,0]
rows = raw_train_ds.shape[0]
train_ds = raw_train_ds.iloc[:, 1:].values.astype(np.float32).reshape((rows, 28,28,1))

#train_ds = np.asarray(train_ds)
#train_ds = train_ds.astype('float').reshape(-1,28,28,1) 
train_ds = torch.from_numpy(train_ds)
train_labels_ds = torch.from_numpy(train_labels_ds.values)

train_ds = torch.reshape(train_ds,(-1,28,28))
train_ds = train_ds.unsqueeze(1)
train_ds = train_ds.unsqueeze(1)
train_labels_ds = torch.reshape(train_labels_ds, (-1,1))

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out

model = CNNModel()

model.cuda()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs=100

iter =0

for epoch in range(num_epochs):
    for i in range(train_ds.shape[0]):
        images = train_ds[i]
        labels = train_labels_ds[i]
        print(images.shape, labels.shape)

        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print("iter=", iter, "loss=",loss.data)
        iter = iter + 1

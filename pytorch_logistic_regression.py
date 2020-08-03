import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

train_datasett = dsets.MNIST(root='./data',train=True, transform=transforms.ToTensor())

test_dataset = dsets.MNIST(root = './data', train=False, transform=transforms.ToTensor())

batch_size = 100

num_epochs = 30

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size. shuffle=False)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.Linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 28*28
output_dim = 10

model = LogisticRegression(input_dim, output_dim)

model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

iter = 0

for epoch in range(num_epochs):


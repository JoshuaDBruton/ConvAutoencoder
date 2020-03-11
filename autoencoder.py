import random

from loader import SnakeGameDataset as loader

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms


class AutoEncoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        # Encoder
        self.enc_layer_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_layer_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4*4*20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)

        # Decoder
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    def encode(self, images):
        code = self.enc_layer_1(images)
        code = F.selu(F.max_pool2d(code, 2))

        code = self.enc_layer_2(code)
        code = F.selu(F.max_pool2d(code, 2))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)

        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = torch.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])

        return out

# Board size
IMAGE_SIZE = 2704
IMAGE_WIDTH = IMAGE_HEIGHT = 52

# Hyper params
code_size = 20
num_epochs = 1
batch_size = 200
lr = 0.002
optimizer_cls = optim.Adam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
# train_data = datasets.MNIST('~/data/mnist/', train=True , transform=transforms.ToTensor(), download=True)
# test_data  = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor(), download=True)
# train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True, pin_memory=True)

train_loader = loader()

autoencoder = AutoEncoder(code_size)
autoencoder.to(device)
loss_fn = nn.BCELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)

    for i, (images, nexts) in enumerate(train_loader.read()):    # Ignore image labels
        images = images.to(device)
        nexts = nexts.to(device)
        out, code = autoencoder(Variable(images))

        optimizer.zero_grad()
        loss = loss_fn(out, nexts)
        loss.backward()
        optimizer.step()

    print("Loss = %.3f" % loss.data)

# Try reconstructing on test data
# test_image = random.choice(test_data)
# test_image = test_image[0]
# test_image = test_image.unsqueeze(0)
# test_image = Variable(test_image.to(device))
# test_reconst, _ = autoencoder(test_image)
#
# torchvision.utils.save_image(test_image, 'orig.png')
# torchvision.utils.save_image(test_reconst, 'reconst.png')

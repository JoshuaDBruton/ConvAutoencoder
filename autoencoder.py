import random

from loader import SnakeGameDataset as dataset

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms

import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        # Encoder
        self.enc_layer_1 = nn.Conv2d(1, 8, kernel_size=9, stride=2)
        self.enc_layer_2 = nn.Conv2d(8, 16, kernel_size=7)
        self.enc_layer_3 = nn.Conv2d(16, 32, kernel_size=5)
        #self.enc_linear_1 = nn.Linear(4*4*20, 50)
        self.enc_linear_1 = nn.Linear(12*12*32, 2300)
        self.enc_linear_2 = nn.Linear(2300, 1000)
        self.enc_linear_3 = nn.Linear(1000, self.code_size)

        # Decoder
        self.dec_linear_1 = nn.Linear(self.code_size, 1000)
        self.dec_linear_2 = nn.Linear(1000, 2300)
        self.dec_linear_3 = nn.Linear(2300, 12*12*32)
        self.dec_layer_1 = nn.ConvTranspose2d(32, 16, kernel_size=5)
        self.dec_layer_2 = nn.ConvTranspose2d(16, 8, kernel_size=7)
        self.dec_layer_3 = nn.ConvTranspose2d(8, 1, kernel_size=9, stride=2, output_padding=1)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    def encode(self, images):
        code = F.selu(self.enc_layer_1(images))

        code = F.selu(self.enc_layer_2(code))

        code = F.selu(self.enc_layer_3(code))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = F.selu(self.enc_linear_2(code))
        code = self.enc_linear_3(code)

        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.selu(self.dec_linear_2(out))
        out = F.selu(self.dec_linear_3(out))
        out = out.view([code.size(0), 32, 12, 12])
        out = F.selu(self.dec_layer_1(out))
        out = F.selu(self.dec_layer_2(out))
        out = F.selu(self.dec_layer_3(out))
        # out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])

        return out

# Board size
IMAGE_SIZE = 2704
IMAGE_WIDTH = IMAGE_HEIGHT = 52

# Hyper params
code_size = 120
num_epochs = 50
batch_size = 64
lr = 0.002
board_num=1
optimizer_cls = optim.Adam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
# train_data = datasets.MNIST('~/data/mnist/', train=True , transform=transforms.ToTensor(), download=True)
# test_data  = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor(), download=True)
# train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True, pin_memory=True)

print("Gathering training data...")
train_data = dataset(root_dir="/home/joshua/Desktop/ConvAutoencoder/data/trainingData", transform=None, prefix="trainingDataBoards1.npy")
print("Gathering testing data...")
test_data = dataset(root_dir="/home/joshua/Desktop/ConvAutoencoder/data/testData", transform=None, prefix="testDataBoards1.npy")
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=7, batch_size=batch_size, drop_last=True, pin_memory=True) # dropped num_workers = 4

# Visualising the trainingData (Only run this with small dataset)
# vis_1, vis_2, vis_3 = random.choice(train_data), random.choice(train_data), random.choice(train_data)
# _, (ax1, ax2, ax3) = plt.subplots(3, 2)
# ax1[0].imshow(vis_1[0])
# ax1[1].imshow(vis_1[1])
# ax2[0].imshow(vis_2[0])
# ax2[1].imshow(vis_2[1])i
# ax3[0].imshow(vis_3[0])
# ax3[1].imshow(vis_3[1])
# plt.savefig("visualisation.png")

print("There are " + str(len(train_data)) + " samples in the training set.\n")

autoencoder = AutoEncoder(code_size)
autoencoder.to(device)
loss_fn = nn.MSELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)

    for i, (images, nexts) in enumerate(train_loader):    # Ignore image labels
        images = images.unsqueeze(1).float()
        images = images.to(device)
        nexts = nexts.unsqueeze(1).float()
        nexts = nexts.to(device)
        out, code = autoencoder(Variable(images))
        optimizer.zero_grad()
        loss = loss_fn(out, nexts)
        loss.backward()
        optimizer.step()

    print("Loss = %.3f" % loss.data)

    if (epoch+1)%5==0:
        # Try reconstructing on test data
        test_image = random.choice(test_data)
        test_image = test_image[0]
        test_image = torch.from_numpy(test_image).unsqueeze(0).unsqueeze(1).float()
        test_image = Variable(test_image.to(device))
        test_reconst, _ = autoencoder(test_image)
        torchvision.utils.save_image(test_image, 'orig.png')
        torchvision.utils.save_image(test_reconst, 'reconst.png')
        torch.save(autoencoder.state_dict(), "model_checkpoint.pth")
        print("\n Next file...")
        board_num=board_num+1
        if board_num == 399:
            board_num = 1
        train_data = dataset(root_dir="/home/joshua/Desktop/ConvAutoencoder/data/trainingData", transform=None, prefix="trainingDataBoards"+str(board_num)+".npy")
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=7, batch_size=batch_size, drop_last=True, pin_memory=True)

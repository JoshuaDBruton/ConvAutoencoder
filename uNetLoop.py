import random

from loader import SnakeGameDataset as dataset

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from   torchvision import datasets, transforms
from unet import UNet

import matplotlib.pyplot as plt

class param:
    board_num = 1
    img_size = (52, 52)
    bs = 8
    num_workers = 4
    lr = 0.001
    epochs = 6
    unet_depth = 500
    unet_start_filters = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = dataset(root_dir="/home/joshua/Desktop/ConvAutoencoder/data/trainingData", transform=None, prefix="trainingDataBoards1.npy")
test_data = dataset(root_dir="/home/joshua/Desktop/ConvAutoencoder/data/testData", transform=None, prefix="testDataBoards1.npy")
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=param.num_workers, batch_size=param.bs, drop_last=True, pin_memory=True)

model = UNet(1, in_channels=1, depth=param.unet_depth, start_filts=param.unet_start_filters, merge_mode='add').to(device)
optim = torch.optim.Adam(model.parameters(), lr=param.lr)

iters = []
train_losses = []
val_losses = []

it = 0

for epoch in range(param.epochs):
    print("Epoch: " + str(epoch))
    for i, (X, y) in enumerate(train_loader):
        X = X.unsqueeze(1).float().to(device)  # [N, 1, H, W]
        y = y.unsqueeze(1).float().to(device)  # [N, H, W] with class indices (0, 1)
        output = model(Variable(X))  # [N, 2, H, W]
        loss = F.mse_loss(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    print("Loss: " + str(loss.data))
    if (epoch+1)%5==0:
        # Update board_num safely
        param.board_num+=1
        if param.board_num==399:
            param.board_num=1
        # Try reconstructing on test data
        test_image = random.choice(test_data)
        test_image = test_image[0]
        test_image = torch.from_numpy(test_image).unsqueeze(0).unsqueeze(1).float()
        test_image = Variable(test_image.to(device))
        test_reconst = model(test_image)
        torchvision.utils.save_image(test_image, 'orig.png')
        torchvision.utils.save_image(test_reconst, 'reconst.png')
        # Train_data updated
        train_data = dataset(root_dir="/home/joshua/Desktop/ConvAutoencoder/data/trainingData", transform=None, prefix="trainingDataBoards" + str(param.board_num) + ".npy")
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, num_workers=param.num_workers, batch_size=param.bs, drop_last=True, pin_memory=True)
        # Save Model
        torch.save(model.state_dict(), "unet_model.pth")

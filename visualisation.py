import numpy as np
import matplotlib.pyplot as plt
import random
from loader import SnakeGameDataset as dataset
from network import AutoEncoder
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

code_size = 120

autoencoder = AutoEncoder(code_size)
autoencoder.load_state_dict(torch.load("/home/joshua/Desktop/ConvAutoencoder/model_checkpoint.pth"))
autoencoder.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)

test_data = dataset(root_dir="/home/joshua/Desktop/ConvAutoencoder/data/testData", transform=None, prefix="testDataBoards1.npy")

vis_1, vis_2, vis_3 = random.choice(test_data), random.choice(test_data), random.choice(test_data)
check_1 = torch.from_numpy(vis_1[0]).unsqueeze(0).unsqueeze(1).float()
check_2 = torch.from_numpy(vis_2[0]).unsqueeze(0).unsqueeze(1).float()
check_3 = torch.from_numpy(vis_3[0]).unsqueeze(0).unsqueeze(1).float()
res_1, res_2, res_3 = autoencoder(Variable(check_1.to(device))), autoencoder(Variable(check_2.to(device))), autoencoder(Variable(check_3.to(device)))
_, (ax1, ax2, ax3) = plt.subplots(3, 3)
ax1[0].imshow(vis_1[0])
ax1[1].imshow(res_1[0][0][0].to("cpu").detach().numpy())
ax1[2].imshow(vis_1[1])
ax2[0].imshow(vis_2[0])
ax2[1].imshow(res_2[0][0][0].to("cpu").detach().numpy())
ax2[2].imshow(vis_2[1])
ax3[0].imshow(vis_3[0])
ax3[1].imshow(res_3[0][0][0].to("cpu").detach().numpy())
ax3[2].imshow(vis_3[1])
plt.savefig("visualisation.png")

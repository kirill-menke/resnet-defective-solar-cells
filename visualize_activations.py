import sys

import torch
import torchvision as tv
import torch.nn as nn
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import model
from trainer import CHECKPOINTS_DIR
from data import ChallengeDataset

# Adapt this to match the model used in train.py
model = tv.models.resnet34(pretrained=True)
model.fc = nn.Sequential(nn.Linear(512, 2), nn.Sigmoid())

# Load weights of given epoch
epoch = int(sys.argv[1])
ckp = torch.load(CHECKPOINTS_DIR + f'/checkpoint_{epoch:03d}.ckp', "cuda")
model.load_state_dict(ckp['state_dict'])
model.eval()

# Load dataset
df = pd.read_csv('data.csv', sep=';')
dataset = ChallengeDataset(df, "val")

# Attach hooks to conv layers
activations = {}
def get_activation(name):
    def conv_hook(model, input, output):
        # Remove batch dimension (=1) and calculate mean of feature maps along the channel dimension
        activations[name] = output.detach().squeeze(0).mean(0).numpy()
    return conv_hook

for module in model.modules():
    if isinstance(module, nn.Conv2d):
        module.register_forward_hook(get_activation(f"Conv {module.in_channels}, {module.out_channels}"))


# Pick sample for which you want to visualize the activations
x, y = dataset[59]
y_pred = model(x.unsqueeze(0)).squeeze()
loss = nn.BCELoss()(y_pred, y)
loss.backward()

# Plot activations
fig = plt.figure(figsize=(30, 50))
for i, (name, feature_map) in enumerate(activations.items()):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(feature_map)
    a.axis("off")
    a.set_title(name, fontsize=30)
plt.savefig('plots/feature_maps.jpg', bbox_inches='tight')
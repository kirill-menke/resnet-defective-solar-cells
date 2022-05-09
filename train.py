import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import model
from data import ChallengeDataset
from trainer import Trainer


# Make results reproducible
t.manual_seed(0)
t.backends.cudnn.benchmark = False
t.backends.cudnn.deterministic = True

# Load the data from the csv file
df = pd.read_csv('data.csv', sep=';')

# Perform a stratified train-test-split
class_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
stratify_labels = [class_map[(x, y)] for x, y in df[['crack', 'inactive']].to_numpy()]

train, test = train_test_split(df, test_size=0.1, shuffle=True, random_state=2, stratify=stratify_labels)
test.reset_index(inplace=True)
train.reset_index(inplace=True)

# Set up data loading for the training and validation
train_dl = t.utils.data.DataLoader(ChallengeDataset(train, "train"), batch_size=16, shuffle=True)
val_dl = t.utils.data.DataLoader(ChallengeDataset(test, "val"), batch_size=1, shuffle=True)

# Create an instance of a pretrained ResNet model
res_net = tv.models.resnet34(pretrained=True)
res_net.fc = nn.Sequential(nn.Linear(512, 2), nn.Sigmoid())

# Optimizer: SGD with Momentum
optim = t.optim.SGD(res_net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

# Learning rate decay
scheduler = t.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 20 , 30 , 40], gamma=0.1)
scheduler2 = t.optim.lr_scheduler.MultiStepLR(optim, milestones=[60, 80, 100, 130], gamma=0.5)

# Loss criterion for multi-class classification
loss = t.nn.BCELoss()

# Start training
trainer = Trainer(res_net, loss, optim, [scheduler, scheduler2], train_dl, val_dl, True)
res = trainer.fit(5)

# Plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.axhline(y = 0.2, color = 'k', linestyle = 'dashed')
plt.axhline(y = 0.1, color = 'k', linestyle = 'dashed')
plt.yscale('log')
ax = plt.gca()
ax.set_ylim([0.01, 1])
plt.legend()
plt.savefig('./plots/losses.png')

import os
import sys

import torch as t

sys.path.append("..")
from trainer import Trainer
from model import *

ONNX_DIR = "../onnx"

if not os.path.exists(ONNX_DIR):
    os.mkdir(ONNX_DIR)

# Adapt this to match the model used in train.py
model = tv.models.resnet34(pretrained=True)
model.fc = nn.Sequential(nn.Linear(512, 2), nn.Sigmoid())

loss = t.nn.BCELoss()
trainer = Trainer(model, loss)

for epoch in sys.argv[1:]:
    trainer.restore_checkpoint(int(epoch))
    trainer.save_onnx(ONNX_DIR + '/checkpoint_{:03d}.onnx'.format(int(epoch)))
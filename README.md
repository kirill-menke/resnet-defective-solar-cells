# Multi-Label Classification of Defective Solar Cells
Within this project I implemented a Residual Neural Network in PyTorch and using it for classifying defects of solar cells.
Solar cells can exhbit various types of degradation caused by inappropriate transportation, installation, or bad weather conditions such as wind, snow, or hail.
The model implemented here, focuses on two different types of defects described in more detail below. 
A ResNet-34 from PyTorch pretrained on the ImageNet database is used for this multi-label classification task.

## Dataset
The training and prediction is performed on electrolumiscence images of functional and defective solar cells.
The original dataset can be found on this [github page](https://github.com/zae-bayern/elpv-dataset) but was labeled differently for the purpose of this project.

Here, the focus lies on two types of defects: 
1. **Cracks**: The size can range from very small cracks (a few pixels) to large cracks that cover the whole cell
2. **Inactive Regions**: These regions are mainly caused by cracks and do not contribute to power production

Accordingly, each row in `data.csv` contains the path to an image and two numbers indicating whether the solar cell has a crack and if an inactive region exists.
The three images below show samples for each type of defect. 
A solar cell can either have a crack, an inactive region, or both with varying degrees of severity:

![crack](doc/cell1108.png) &nbsp;&nbsp; ![inactive](doc/cell1623.png) &nbsp;&nbsp; ![crack](doc/cell0376.png)

In total the `images/` folder contains 12,000 samples of 300x300 pixels 8-bit grayscale images. 
However, only the first 2,000 samples are unique and the rest was created through data augmentation.
Each image was flipped horizontally and vertically and rotated three times by 90 degrees, meaning 6 different variations were created for each sample.

The original 2,000 samples have the following data distribution:
- *Functional Cells*: 1545
- *Cells with Cracks*: 443
- *Cells with inactive regions*: 122
- *Cells with both defects*: 110

## Implementation
The code snippets below can be found in `train.py` and briefly describes the process and implementation of the model.

Initially, the data is split into training and test set. To ensure that the samples are equally distributed between both sets, a stratified split is performed:
```python
# Load the data from the csv file
df = pd.read_csv('data.csv', sep=';')

# Perform a stratified train-test-split
class_map = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
stratify_labels = [class_map[(x, y)] for x, y in df[['crack', 'inactive']].to_numpy()]
train, test = train_test_split(df, test_size=0.1, shuffle=True, random_state=2, stratify=stratify_labels)
```

For the classification the ResNet-34 of PyTorch was used, which was already pre-trained on the ImageNet database. 
The original final linear layer is replaced by a linear layer with only two output neurons, since we work with only two classes (Cracks and Inactive regions) compared to the ImageNet dataset:
```python
# Create an instance of a pretrained ResNet model
res_net = tv.models.resnet34(pretrained=True)
res_net.fc = nn.Sequential(nn.Linear(512, 2), nn.Sigmoid())
```
An alternative custom ResNet has been defined in `model.py` and can also be used here instead.
However, a comparison has shown that the pre-trained variant performs better (i.e. has a higher f1-score).

A simple SGD with momentum is used as loss function:
```python
# Optimizer: SGD with Momentum
optim = t.optim.SGD(res_net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
```

The Binary Cross Entropy loss is used sind a solar cell can have both defects, i.e. the classes are not mutually exclusive.
```python
# Loss criterion for multi-label classification
loss = t.nn.BCELoss()
```


## Usage

- visualize_activations.py
- export_onnx



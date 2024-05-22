# CheckpointX
## Install
```
pip install -r requirements.txt
```
For running examples
```
pip install -r requirements_example.txt
```

## General Usage
Use ```CheckpointX.utils.checkpoint.CheckpointXRunner()``` and ```checkpointx_runner.checkpointx_sequential(...)```
to replace ```torch.utils.checkpoint.checkpoint_sequential(...)```

For example
```
import torch.nn as nn
import torch.optim as optim
import CheckpointX

net = nn.Seqeuntial(...) # define model as a sequential
checkpointx_runner = CheckpointX.utils.checkpoint.CheckpointXRunner()
inputs = ... # define your network inputs
net = net.cuda() # send to gpu
optimizer = optim.Adam(net.parameters(), lr=1e-04)
# For training loop
for data in dataloader:
    optimizer.zero_grad()
    inputs, targets = data
    inputs, targets = inputs.cuda(), targets.cuda()
    # for example, reduce memory to 50%
    # the first invokation will run checkpointx solver to solve for optimal checkpointing scheme
    outputs = checkpointx_runner.checkpoint_sequential(net, inputs, fraction=0.5)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
```

## Examples
We provide MLP, Resnet, and ViT example scripts under example/
import torch
import torch.nn as nn

# Example target tensor with class labels (Long data type)
target = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

# Example input tensor with class scores or logits (Float data type)
input = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7]
], dtype=torch.float)

# Calculate the cross-entropy loss
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(input, target)
print(loss)
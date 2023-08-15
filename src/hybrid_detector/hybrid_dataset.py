import torch
from torch.utils.data import Dataset

class HybridDataset(Dataset):
  def __init__(self, dataset, transform_list=None):
    [data_X, data_y] = dataset
    X_tensor, y_tensor = torch.tensor(data_X), torch.tensor(data_y)
    tensors = (X_tensor, y_tensor)
    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    self.tensors = tensors
    self.transforms = transform_list

  def __getitem__(self, index):
    x = self.tensors[0][index]

    if self.transforms:
      x = self.transforms(x)

    y = self.tensors[1][index]

    return x, y

  def __len__(self):
    return self.tensors[0].size(0)
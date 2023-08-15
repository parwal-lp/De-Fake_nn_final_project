import torch
from torch.utils.data import Dataset, DataLoader

from encoder import encode_images_and_captions, fuse_embeddings

class HybridDataset(Dataset):
    def __init__(self, fused_img_captions, labels, transform_list=None):
        self.data = []
        self.transforms = transform_list

        for sample, label in zip(fused_img_captions, labels):
            self.data.append([sample, label])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        sample, class_name = self.data[index]
        return sample, class_name
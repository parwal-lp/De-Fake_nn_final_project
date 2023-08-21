from torch.utils.data import Dataset

class HybridDataset(Dataset):
    def __init__(self, fused_img_captions, labels, transform_list=None):
        self.data = fused_img_captions
        self.labels = labels
        self.transforms = transform_list

        #for sample, label in zip(fused_img_captions, labels):
        #    self.data.append([sample, label])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        sample, class_name = self.data[index], self.labels[index]
        return sample, class_name
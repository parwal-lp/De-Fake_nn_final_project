import torch
import torch.nn as nn
import os
import clip
import pandas as pd
import torch.nn.functional as F

import pandas as pd
import torch
import os
import clip
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



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

def fuse_embeddings(encoded_images, encoded_labels):
    fused_embeddings = []
    for img, lab in zip(encoded_images, encoded_labels):
        img_lab = torch.cat((img, lab), dim=1)
        fused_embeddings.append(img_lab)
    return fused_embeddings

def encode_multiclass_images_and_captions(captions_file, dataset_dir, class_names):

    clip_dir = "../CLIP/"
    os.chdir(clip_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    os.chdir("../De-Fake_nn_final_project")
    #print(os.getcwd())

    df = pd.read_csv(captions_file)

    encoded_images = []
    encoded_captions = []
    labels = []

    #create encodings for images and texts
    for class_name in class_names:
        path_to_class = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(path_to_class): #scorro prima le real e poi le fake
            #print(dir, img_name)
            img_path = os.path.join(path_to_class, img_name)
            img = Image.open(img_path)

            #tensor_img = transforms.ToTensor()(img).to(device)
            tensor_img = preprocess(img).unsqueeze(0).to(device)
            encoded_img = model.encode_image(tensor_img)

            caption = ""
            img_class = None
            for index, row in df.iterrows():
                if (class_name == "class_real" or class_name == "class_1"): #le real hanno il nome uguale all'id
                    if str(df.iloc[index]['img_id']) == img_name[:-4]:
                        caption = df.iloc[index]['caption']
                        # img_class = 1
                else: #le fake hanno il nome uguale a "fake_id", defo rimuovere il "fake_"
                    if str(df.iloc[index]['img_id']) == img_name[5:-4]:
                        caption = df.iloc[index]['caption']
                        #img_class = 0

                #img_class = [class_name=="class_real", class_name=="class_SD", class_name=="class_LD", class_name=="class_GLIDE"]
                if class_name == "class_real": img_class = 0
                elif class_name == "class_SD": img_class = 1
                elif class_name == "class_LD": img_class = 2
                elif class_name == "class_GLIDE": img_class = 3


            #tensor_label = transforms.ToTensor()(label).to(device)
            tensor_caption = torch.cat([clip.tokenize(caption)]).to(device)
            encoded_caption = model.encode_text(tensor_caption)

            encoded_images.append(encoded_img)
            encoded_captions.append(encoded_caption)
            labels.append(img_class)

    #print(labels)
    return encoded_images, encoded_captions, labels


def get_multiclass_dataset_loader(captions_file, dataset_dir, classes):
    imgs, captions, labels = encode_multiclass_images_and_captions(captions_file, dataset_dir, classes)
    fused_imgs_captions = fuse_embeddings(imgs, captions)

    fused_imgs_captions = torch.stack(fused_imgs_captions).to(torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    hybrid_dataset = HybridDataset(fused_imgs_captions, labels)
    data_loader = DataLoader(hybrid_dataset, batch_size=5, shuffle=True)

    return data_loader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("loading dataset...")
captions_file = "data/hybrid_attributor_data/train/mscoco_captions.csv"
dataset_dir = "data/hybrid_attributor_data/train"
classes = {"class_real", "class_SD", "class_LD", "class_GLIDE"}

trainloader = get_multiclass_dataset_loader(captions_file, dataset_dir, classes)

net = Net().to(device)

import torch.optim as optim
print("training model...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (sample_data, sample_class) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = sample_data.to(device)
        labels = sample_class.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        outputs = net(inputs)
        
        outputs = outputs.squeeze()
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
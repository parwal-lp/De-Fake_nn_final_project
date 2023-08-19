import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import torchvision

from torch.optim.lr_scheduler import CosineAnnealingLR

cudnn.benchmark = True
plt.ion()   # interactive mode


## ---------------- PREPARE THE DATA -------------------- ##
data_transforms = {
    'train': torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'data/imageonly_attributor_data'
image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## ------------------- DEFINE TRAIN FUNCTION ---------------- ##

def train_model(model, criterion, optimizer, num_epochs):

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluation/testing mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        #print(f'inputs: {inputs}')
                        outputs = model(inputs)

                        

                        #print(f'outputs: {outputs}')
                        #print(f'labels: {labels}')
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        #print(f"PREDICTION: {preds} - LABELS: {labels}")
                        #print(f'preds: {preds}')
                        #print(f'loss: {loss}')

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # resetta i gradienti
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def eval_model(model, criterion):
    model.eval()

    for phase in ['test']:

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            with torch.set_grad_enabled(False):
                #print(f'inputs: {inputs}')
                outputs = model(inputs)
                #print(f'outputs: {outputs}')
                #print(f'labels: {labels}')
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        loss = running_loss / dataset_sizes[phase]
        acc = running_corrects.double() / dataset_sizes[phase]

        print(f'TEST LOSS: {loss:.4f} ACC: {acc:.4f}')

## ---------------------- DEFINE MODEL -------------------------- ##
model_ft = torchvision.models.resnet18(weights='IMAGENET1K_V1')

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(),lr=1e-4)

## START TRAINING THE MODEL ##
model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=50)

#eval_model(model_ft, criterion)
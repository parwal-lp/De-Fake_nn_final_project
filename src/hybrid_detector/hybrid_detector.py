import os
import time
import torch
import torch.nn as nn

from torch.autograd import Variable

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

class TwoLayerPerceptron(torch.nn.Module):
    # 2-layer multilayer perceptron
    # to be used for hybrid detection

    def __init__(self, in_size, h_size, out_size):
        super(TwoLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h_size, out_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train_test(model, criterion, optimizer, scheduler, num_epochs, dataloader):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # resetta i gradienti
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    #print(f'inputs: {inputs}')
                    outputs = model(inputs)
                    outputs = outputs.squeeze()
                    #print(f'outputs: {outputs}')
                    #print(f'labels: {labels}')
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    #print(f'preds: {preds}')
                    #print(f'loss: {loss}')

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(outputs == labels.data)
            
            scheduler.step()

            epoch_loss = running_loss / 100
            epoch_acc = running_corrects.double() / 100

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def train_hybrid_detector(model, data_loader, epochs=50, learning_rate=0.05):
    model.train()

    for epoch in range(0, epochs):
        #define loss function
        loss_fn = nn.BCELoss()
        #define optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        losses = []
        accuracies = []

        print(f"epoch: {epoch}/{epochs}")

        for i, (data, label) in enumerate(data_loader):
            print(f"batch: {i}")
            
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                pred = model(data)
                pred = pred.squeeze()

                loss = loss_fn(pred, label)

                losses.append(loss)
            
                loss.backward(retain_graph=True)
                optimizer.step()
            
            with torch.no_grad():
                acc = (pred.round() == label).float().mean()
                accuracies.append(acc)


            print(f'batch {i}/{data_loader.__len__()} - acc: {acc}')

        print("EPOCH: ", epoch, " - MEAN ACCURACY: ", torch.mean(torch.tensor(accuracies)), " - MEAN LOSS: ", torch.mean(torch.tensor(losses)))

    torch.save(model.state_dict(), 'trained_models/hybrid_detector.pth')

def eval_hybrid_detector(model, data_loader):
    model.eval()
    #define loss function
    loss_fn = nn.BCELoss()
    
    losses = []
    accuracies = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for data, label in data_loader:
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = model(data)
            pred = pred.squeeze()

            loss = loss_fn(pred, label.float())
            
            acc = (pred.round() == label).float().mean()
            # print(f'{pred} == {label} --> ACC: {acc} LOSS: {loss}')

            losses.append(loss)
            accuracies.append(acc)

    mean_loss = torch.mean(torch.tensor(losses))
    mean_acc = torch.mean(torch.tensor(accuracies))
    return mean_loss, mean_acc
import torch
import torch.nn as nn

import pandas as pd
import torch

class MultiClassTwoLayerPerceptron(torch.nn.Module):
    # 2-layer multilayer perceptron
    # to be used for hybrid attribution

    def __init__(self, in_size, h_size, out_size):
        super(MultiClassTwoLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h_size, out_size)
        
    def forward(self, inputs):
        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_hybrid_attributor(model, data_loader, epochs, learning_rate):
    model.train()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define loss function
    loss_fn = nn.CrossEntropyLoss()
    #define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(0, epochs):

        losses = []
        accuracies = []

        for i, (sample_data, sample_class) in enumerate(data_loader):
            
            data = sample_data.to(device)
            label = sample_class.to(device)

            optimizer.zero_grad()

            #with torch.set_grad_enabled(True):
            pred = model(data)
            
            #pred = pred.squeeze()

            

            loss = loss_fn(pred, label)
            
            losses.append(loss)
            
            loss.backward()
            optimizer.step()

            
            predicted_class = torch.argmax(pred, dim=1)
            #print(f"prediction: {predicted_class}, real label: {label}")
        
            acc = (predicted_class == label).float().mean()
            accuracies.append(acc)


            #print(f'epoch {epoch+1}, batch {i+1}/{data_loader.__len__()} - acc: {acc}')

        print("EPOCH: ", f"{epoch+1}/{epochs}", " - MEAN ACCURACY: ", torch.mean(torch.tensor(accuracies)), " - MEAN LOSS: ", torch.mean(torch.tensor(losses)))

    torch.save(model.state_dict(), 'trained_models/hybrid_attributor.pth')
import torch
import torch.nn as nn

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


def train_hybrid_detector(model, data_loader, epochs=50, learning_rate=0.05):
    model.train()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(0, epochs):
        
        losses = []
        accuracies = []

        for i, (data, label) in enumerate(data_loader):
            
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                
                pred = model(data)
                #print(pred.shape)
                pred = pred.squeeze()

                loss = loss_fn(pred, label)

                losses.append(loss)
            
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                acc = (pred.round() == label).float().mean()
                accuracies.append(acc)


            #print(f'batch {i}/{data_loader.__len__()} - acc: {acc}')

        print("EPOCH: ", f"{epoch+1}/{epochs}", " - MEAN ACCURACY: ", torch.mean(torch.tensor(accuracies)), " - MEAN LOSS: ", torch.mean(torch.tensor(losses)))

    torch.save(model.state_dict(), 'trained_models/hybrid_detector.pth')

def eval_hybrid_detector(model, data_loader):
    model.eval()
    #define loss function
    loss_fn = nn.BCELoss()
    
    losses = []
    accuracies = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
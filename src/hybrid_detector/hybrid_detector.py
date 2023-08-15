import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from encoder import encode_images_and_captions, fuse_embeddings
from hybrid_dataset import HybridDataset

class TwoLayerPerceptron(torch.nn.Module):
    # 2-layer multilayer perceptron
    # to be used for hybrid detection

    def __init__(self, in_size, h_size, out_size):
        super(TwoLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h_size, out_size)
        self.sigmoid = nn.Sigmoid()

        self.W1 = nn.Parameter(torch.empty(in_size, h_size))
        self.b1 = nn.Parameter(torch.empty((h_size,1)))
        self.W2 = nn.Parameter(torch.empty(h_size, out_size))
        self.b2 = nn.Parameter(torch.empty((out_size,1)))

        #W1 = np.random.randn(h_size,in_size) * 0.01  # random initialization
        #b1 = np.zeros((h_size,1))
        #W2 = np.random.randn(out_size,h_size) * 0.01  # random initialization
        #b2 = np.zeros((out_size,1))

        nn.init.normal_(self.W1, 0, 0.01)
        nn.init.constant_(self.b1, 0.0)
        nn.init.normal_(self.W2, 0, 0.01)
        nn.init.constant_(self.b2, 0.0)
        
    def forward(self, inputs):
        #W1 = self.parameters['W1']   #(n_x,n_h)
        #b1 = self.parameters['b1']   #(1,n_h)
        #W2 = self.parameters['W2']   #(n_h,n_y)
        #b2 = self.parameters['b2']   #(1,n_y)

        W1 = self.W1
        b1 = self.b1
        W2 = self.W2
        b2 = self.b2

        #Z1 = torch.dot(W1,inputs) + b1
        #A1 = torch.tanh(Z1) #apply tanh activation
        #Z2 = torch.dot(W2, A1) + b2
        #A2 = 1/(1+torch.exp(-Z2)) #apply sigmoid activation   #(n_y,m)

        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out
    
def train_hybrid_detector(model, Xtrain, ytrain, epochs=50, learning_rate=0.05):
    model.train()
    #define loss function
    loss_fn = nn.BCELoss()
    #define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []

    for epoch in range(0, epochs):   

        for data, label in zip(Xtrain,ytrain):
            
            data = data.to(device)
            label = label.to(device)
            print(f'label: {label}')

            optimizer.zero_grad() #set gradients to zero

            with torch.set_grad_enabled(True):
                pred = model(data)
                pred = pred.squeeze()

                print(f'prediction: {pred}')

                loss = loss_fn(pred, label)
                acc = (pred.round() == label).float().mean()

                losses.append(loss)
                accuracies.append(acc)

                print(f'accuracy: {acc}')
        
                loss.backward(retain_graph=True)
                optimizer.step()

        print("EPOCH: ", epoch, " - MEAN ACCURACY: ", torch.mean(torch.tensor(accuracies)), " - MEAN LOSS: ", torch.mean(torch.tensor(losses)))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hybrid_detector = TwoLayerPerceptron(1024, 100, 1).to(device)

#build train dataset
captions_file = "data/hybrid_detector_data/mscoco_captions.csv"
real_img_dir = "data/hybrid_detector_data/train/class_1"
fake_img_dir = "data/hybrid_detector_data/train/class_0"

imgs, captions, labels = encode_images_and_captions(captions_file, real_img_dir, fake_img_dir)
fused_imgs_captions = fuse_embeddings(imgs, captions)

fused_imgs_captions = torch.stack(fused_imgs_captions).to(torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

#dataset_train = HybridDataset(dataset=fused_imgs_captions)

#trainloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

train_hybrid_detector(hybrid_detector, fused_imgs_captions, labels, 100, 0.05)

    #do predictions
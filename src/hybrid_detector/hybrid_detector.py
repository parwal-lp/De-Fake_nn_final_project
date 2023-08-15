import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
        
    def forward(self, inputs):
        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
def train_hybrid_detector(model, data_loader, epochs=50, learning_rate=0.05):
    model.train()
    #define loss function
    loss_fn = nn.BCELoss()
    #define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(0, epochs):   
        losses = []
        accuracies = []

        #for data, label in zip(Xtrain,ytrain):
        i = 0
        for data, label in data_loader:
            i += 1
            
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad() #set gradients to zero

            with torch.set_grad_enabled(True):
                pred = model(data)
                pred = pred.squeeze()

                loss = loss_fn(pred, label)
                acc = (pred.round() == label).float().mean()

                losses.append(loss)
                accuracies.append(acc)
        
                loss.backward(retain_graph=True)
                optimizer.step()

            print(f'batch {i}/{data_loader.__len__()} - acc: {acc}')

        print("EPOCH: ", epoch, " - MEAN ACCURACY: ", torch.mean(torch.tensor(accuracies)), " - MEAN LOSS: ", torch.mean(torch.tensor(losses)))

    torch.save(model.state_dict(), 'hybrid_detector.pth')

def eval_hybrid_detector(model, data_loader):
    model.eval()
    #define loss function
    loss_fn = nn.BCELoss()
    
    losses = []
    accuracies = []

    for data, label in data_loader:
        data = data.to(device)
        label = label.to(device)

        with torch.set_grad_enabled(True):
            pred = model(data)
            pred = pred.squeeze()

            loss = loss_fn(pred, label)
            acc = (pred.round() == label).float().mean()

            losses.append(loss)
            accuracies.append(acc)

    mean_loss = torch.mean(torch.tensor(losses))
    mean_acc = torch.mean(torch.tensor(accuracies))
    return mean_loss, mean_acc

## -------- TRAIN --------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hybrid_detector = TwoLayerPerceptron(1024, 100, 1).to(device)

# -----------------------------------------  build train dataset
# captions_file = "data/hybrid_detector_data/mscoco_captions.csv"
# real_img_dir = "data/hybrid_detector_data/train/class_1"
# fake_img_dir = "data/hybrid_detector_data/train/class_0"

# imgs, captions, labels = encode_images_and_captions(captions_file, real_img_dir, fake_img_dir)
# fused_imgs_captions = fuse_embeddings(imgs, captions)

# fused_imgs_captions = torch.stack(fused_imgs_captions).to(torch.float32)
# labels = torch.tensor(labels, dtype=torch.float32)

# hybrid_dataset = HybridDataset(fused_imgs_captions, labels)
# data_loader = DataLoader(hybrid_dataset, batch_size=5, shuffle=True)

# train_hybrid_detector(hybrid_detector, data_loader, 25, 0.05)


## -------- EVAL --------------------------------------------------------------
# TODO riscrivi tutte ste righe orribili in un unico for che valuta tuti i dataset
# TODO capisci bene coem si calcola accuracy e loss in caso di batch (mi pare che vengono valori poco sensati, la somma di loss e acc non deve fare 1?)
os.chdir("../De-Fake_nn_final_project")
test_hybrid_detector = TwoLayerPerceptron(1024, 100, 1).to(device)
test_hybrid_detector.load_state_dict(torch.load('hybrid_detector.pth'))

# ------------------------------------------  build test dataset SD
SD_captions_file = "data/hybrid_detector_data/mscoco_captions.csv"
SD_real_img_dir = "data/hybrid_detector_data/val/class_1"
SD_fake_img_dir = "data/hybrid_detector_data/val/class_0"

SD_imgs, SD_captions, SD_labels = encode_images_and_captions(SD_captions_file, SD_real_img_dir, SD_fake_img_dir)
SD_fused_imgs_captions = fuse_embeddings(SD_imgs, SD_captions)

SD_fused_imgs_captions = torch.stack(SD_fused_imgs_captions).to(torch.float32)
SD_labels = torch.tensor(SD_labels, dtype=torch.float32)

SD_hybrid_dataset = HybridDataset(SD_fused_imgs_captions, SD_labels)
SD_data_loader = DataLoader(SD_hybrid_dataset, batch_size=5, shuffle=True)

SDloss, SDacc = eval_hybrid_detector(test_hybrid_detector, SD_data_loader)
print(f'Evaluation on SD --> Accuracy: {SDacc} Loss: {SDloss}')

# ------------------------------------------  build test dataset GLIDE
SD_captions_file = "data/hybrid_detector_data/val_GLIDE/mscoco_captions.csv"
SD_real_img_dir = "data/hybrid_detector_data/val_GLIDE/class_1"
SD_fake_img_dir = "data/hybrid_detector_data/val_GLIDE/class_0"

SD_imgs, SD_captions, SD_labels = encode_images_and_captions(SD_captions_file, SD_real_img_dir, SD_fake_img_dir)
SD_fused_imgs_captions = fuse_embeddings(SD_imgs, SD_captions)

SD_fused_imgs_captions = torch.stack(SD_fused_imgs_captions).to(torch.float32)
SD_labels = torch.tensor(SD_labels, dtype=torch.float32)

SD_hybrid_dataset = HybridDataset(SD_fused_imgs_captions, SD_labels)
SD_data_loader = DataLoader(SD_hybrid_dataset, batch_size=5, shuffle=True)

SDloss, SDacc = eval_hybrid_detector(test_hybrid_detector, SD_data_loader)
print(f'Evaluation on GLIDE --> Accuracy: {SDacc} Loss: {SDloss}')

# ------------------------------------------  build test dataset LD
SD_captions_file = "data/hybrid_detector_data/val_LD/mscoco_captions.csv"
SD_real_img_dir = "data/hybrid_detector_data/val_LD/class_1"
SD_fake_img_dir = "data/hybrid_detector_data/val_LD/class_0"

SD_imgs, SD_captions, SD_labels = encode_images_and_captions(SD_captions_file, SD_real_img_dir, SD_fake_img_dir)
SD_fused_imgs_captions = fuse_embeddings(SD_imgs, SD_captions)

SD_fused_imgs_captions = torch.stack(SD_fused_imgs_captions).to(torch.float32)
SD_labels = torch.tensor(SD_labels, dtype=torch.float32)

SD_hybrid_dataset = HybridDataset(SD_fused_imgs_captions, SD_labels)
SD_data_loader = DataLoader(SD_hybrid_dataset, batch_size=5, shuffle=True)

SDloss, SDacc = eval_hybrid_detector(test_hybrid_detector, SD_data_loader)
print(f'Evaluation on LD --> Accuracy: {SDacc} Loss: {SDloss}')
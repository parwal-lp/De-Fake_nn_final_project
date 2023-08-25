import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
from tempfile import TemporaryDirectory

def train_imageonly_detector(model, dataloaders, dataset_sizes, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    since = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
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

                    # resetta i gradienti
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        #print(f'inputs: {inputs}')
                        outputs = model(inputs)
                        #print(f'outputs: {outputs}')
                        #print(f'labels: {labels}')
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        #print(f'preds: {preds}')
                        #print(f'loss: {loss}')

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    
    if not os.path.isdir("trained_models/"): os.makedirs("trained_models/")

    torch.save(model.state_dict(), 'trained_models/imageonly_detector.pth')
    return model


def eval_imageonly_detector(model, dataloaders, dataset_sizes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    since = time.time()

    for phase in ['val', 'val_LD', 'val_GLIDE']:
        if phase == 'val': dataset_name = 'SD'
        if phase == 'val_LD': dataset_name = 'LD'
        if phase == 'val_GLIDE': dataset_name = 'GLIDE'

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

        print(f'Evaluation on {dataset_name} -> Acc: {acc:.4f} - Loss: {loss:.4f}')

    time_elapsed = time.time() - since
    print(f'Evaluation complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

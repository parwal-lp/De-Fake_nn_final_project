
import torch
from tempfile import TemporaryDirectory
import os
import torch.nn as nn
import torch.optim as optim



def train_imageonly_attributor(model, dataloaders, dataset_sizes, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-4)

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


def eval_imageonly_attributor(model, dataloaders, dataset_sizes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    criterion = nn.CrossEntropyLoss()

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
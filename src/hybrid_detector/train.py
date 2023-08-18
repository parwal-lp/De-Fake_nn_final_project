import torch
from encoder import get_dataset_loader
from hybrid_detector import TwoLayerPerceptron, train_hybrid_detector, train_test

from torch.optim import lr_scheduler




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Building the model...')
hybrid_detector = TwoLayerPerceptron(1024, 100, 1).to(device)

print('Building the dataset...')
captions_file = "data/hybrid_detector_data/mscoco_captions.csv"
real_img_dir = "data/hybrid_detector_data/train/class_1"
fake_img_dir = "data/hybrid_detector_data/train/class_0"
train_data_loader = get_dataset_loader(captions_file, real_img_dir, fake_img_dir)

print('Training starts:')
train_hybrid_detector(hybrid_detector, train_data_loader, 5, 0.05)
#define loss function
#loss_fn = torch.nn.BCELoss()
#define optimizer
#optimizer = torch.optim.SGD(hybrid_detector.parameters(), lr=0.05)
#train(hybrid_detector, optimizer, train_data_loader, loss_fn, 5)

#scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#train_test(hybrid_detector, loss_fn, optimizer, scheduler, 5, train_data_loader)

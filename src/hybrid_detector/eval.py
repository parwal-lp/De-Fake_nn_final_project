import torch

from hybrid_detector import TwoLayerPerceptron, eval_hybrid_detector
from encoder import get_dataset_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("loading model with trained weights...")
test_hybrid_detector = TwoLayerPerceptron(1024, 100, 1).to(device)
test_hybrid_detector.load_state_dict(torch.load('trained_models/hybrid_detector.pth'))

print("STARTING EVALUATION")

eval_dirs = {'SD': {
                'captions': "data/hybrid_detector_data/mscoco_captions.csv", 
                'real': "data/hybrid_detector_data/val/class_1", 
                'fake': "data/hybrid_detector_data/val/class_0"},
             'GLIDE': {
                 'captions': "data/hybrid_detector_data/val_GLIDE/mscoco_captions.csv",
                  'real': "data/hybrid_detector_data/val_GLIDE/class_1", 
                  'fake': "data/hybrid_detector_data/val_GLIDE/class_0"},
             'LD': {
                 'captions': "data/hybrid_detector_data/val_LD/mscoco_captions.csv", 
                 'real': "data/hybrid_detector_data/val_LD/class_1", 
                 'fake': "data/hybrid_detector_data/val_LD/class_0"}}

for dataset_name in eval_dirs:
    eval_data_loader = get_dataset_loader(eval_dirs[dataset_name]['captions'], eval_dirs[dataset_name]['real'], eval_dirs[dataset_name]['fake'])
    SDloss, SDacc = eval_hybrid_detector(test_hybrid_detector, eval_data_loader)
    print(f'Evaluation on {dataset_name} --> Accuracy: {SDacc} - Loss: {SDloss}')
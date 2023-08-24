import pandas as pd
import torch
import os
import clip
from PIL import Image
from torch.utils.data import DataLoader

from src.hybrid_dataset import HybridDataset
from torch.utils.data import TensorDataset

# these instructions are needed to:
# - encode an input image using CLIP image encoder
# - encode its label using CLIP text encoder
# - concatenate these two embeddings into a single item (that becomes the input for the training of our classifier)

def encode_images_and_captions(captions_file, real_img_dir, fake_img_dir, clip_dir, proj_dir):

    #clip_dir = "../CLIP/"
    os.chdir(clip_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    #proj_dir = "../De-Fake_nn_final_project"
    os.chdir(proj_dir)
    #print(os.getcwd())

    img_dirs = [real_img_dir, fake_img_dir]
    df = pd.read_csv(captions_file)

    encoded_images = []
    encoded_captions = []
    labels = []

    #create encodings for images and texts
    for dir in img_dirs:
        for img_name in os.listdir(dir): #scorro prima le real e poi le fake
            #print(dir, img_name)
            img_path = os.path.join(dir, img_name)
            img = Image.open(img_path)

            #tensor_img = transforms.ToTensor()(img).to(device)
            tensor_img = preprocess(img).unsqueeze(0).to(device)
            encoded_img = model.encode_image(tensor_img)
            encoded_img = encoded_img.squeeze()

            caption = ""
            img_class = None
            for index, row in df.iterrows():
                if (dir == real_img_dir): #le real hanno il nome uguale all'id
                    if str(df.iloc[index]['img_id']) == img_name[:-4]:
                        caption = df.iloc[index]['caption']
                        img_class = 1
                elif (dir == fake_img_dir): #le fake hanno il nome uguale a "fake_id", defo rimuovere il "fake_"
                    if str(df.iloc[index]['img_id']) == img_name[5:-4]:
                        caption = df.iloc[index]['caption']
                        img_class = 0


            #tensor_label = transforms.ToTensor()(label).to(device)
            tensor_caption = clip.tokenize(caption).to(device)
            encoded_caption = model.encode_text(tensor_caption)
            encoded_caption = encoded_caption.squeeze()

            encoded_images.append(encoded_img)
            encoded_captions.append(encoded_caption)
            labels.append(img_class)

    #print(labels)
    return encoded_images, encoded_captions, labels


def fuse_multiclass_embeddings(encoded_images, encoded_labels):
    fused_embeddings = []
    for img, lab in zip(encoded_images, encoded_labels):
        img_lab = torch.cat((img, lab))
        #print(img_lab.shape)
        fused_embeddings.append(img_lab)
    return fused_embeddings

def fuse_embeddings(encoded_images, encoded_labels):
    fused_embeddings = []
    for img, lab in zip(encoded_images, encoded_labels):
        img_lab = torch.cat((img, lab))#, dim=1)
        fused_embeddings.append(img_lab)
    return fused_embeddings

def get_dataset_loader(captions_file, real_img_dir, fake_img_dir, clip_dir, proj_dir):

    imgs, captions, labels = encode_images_and_captions(captions_file, real_img_dir, fake_img_dir, clip_dir, proj_dir)
    fused_imgs_captions = fuse_embeddings(imgs, captions)

    fused_imgs_captions = torch.stack(fused_imgs_captions).to(torch.float32)
    for tensor in fused_imgs_captions:
        tensor = tensor.detach()
    fused_imgs_captions = fused_imgs_captions.detach()

    labels = torch.tensor(labels, dtype=torch.float32)

    hybrid_dataset = TensorDataset(fused_imgs_captions, labels)
    data_loader = DataLoader(hybrid_dataset, batch_size=5, shuffle=True)

    return data_loader


def encode_multiclass_images_and_captions(captions_file, dataset_dir, class_names, clip_dir, proj_dir):

    #clip_dir = "../CLIP/"
    os.chdir(clip_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    #proj_dir = "../De-Fake_nn_final_project"
    os.chdir(proj_dir)
    #print(os.getcwd())

    df = pd.read_csv(captions_file)

    encoded_images = []
    encoded_captions = []
    labels = []

    #create encodings for images and texts
    for class_name in class_names:
        path_to_class = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(path_to_class): #scorro prima le real e poi le fake
            #print(dir, img_name)
            img_path = os.path.join(path_to_class, img_name)
            img = Image.open(img_path)

            #tensor_img = transforms.ToTensor()(img).to(device)
            tensor_img = preprocess(img).unsqueeze(0).to(device) # Ritorna un tensore batchsize*channels*height*width (1x3x224x224)
            
            encoded_img = model.encode_image(tensor_img) # Ritorna un tensore batchsize*num_features (1x512)
            encoded_img = encoded_img.squeeze() # rimuovo la dimensione della batchsize=1
            #print(encoded_img.shape)
            caption = ""
            img_class = None
            for index, row in df.iterrows():
                if (class_name == "class_real" or class_name == "class_1"): #le real hanno il nome uguale all'id
                    if str(df.iloc[index]['img_id']) == img_name[:-4]:
                        caption = df.iloc[index]['caption']
                        # img_class = 1
                else: #le fake hanno il nome uguale a "fake_id", defo rimuovere il "fake_"
                    if str(df.iloc[index]['img_id']) == img_name[5:-4]:
                        caption = df.iloc[index]['caption']
                        #img_class = 0

                #img_class = [class_name=="class_real", class_name=="class_SD", class_name=="class_LD", class_name=="class_GLIDE"]
                if class_name == "class_real": img_class = 0
                elif class_name == "class_SD": img_class = 1
                elif class_name == "class_LD": img_class = 2
                elif class_name == "class_GLIDE": img_class = 3


            #tensor_label = transforms.ToTensor()(label).to(device)
            #tensor_caption = torch.cat([clip.tokenize(caption)]).to(device)
            tensor_caption = clip.tokenize(caption).to(device) # Returns a tensor batchsize*num_tokens (1x77)
            
            encoded_caption = model.encode_text(tensor_caption) # Returns a tensor batchsize*num_features (1x512)
            encoded_caption = encoded_caption.squeeze()
            #print(encoded_caption.shape)
            encoded_images.append(encoded_img)
            encoded_captions.append(encoded_caption)
            labels.append(img_class)

    #print(labels)
    return encoded_images, encoded_captions, labels


def get_multiclass_dataset_loader(captions_file, dataset_dir, classes, clip_dir, proj_dir):
    imgs, captions, labels = encode_multiclass_images_and_captions(captions_file, dataset_dir, classes, clip_dir, proj_dir) # questa funzione restituisce delle liste
    #print(len(imgs), len(captions), len(labels))
    fused_imgs_captions = fuse_multiclass_embeddings(imgs, captions) # questa funzione ritorna una lista

    #print("embedding created by my function: ", fused_imgs_captions[0])

    fused_imgs_captions = torch.stack(fused_imgs_captions).to(torch.float32) # ottengo un unico tensore che contiene tutti gli embeddings (n_embeddings*dim_embeddings = 198x1024)
    for tensor in fused_imgs_captions:
        tensor = tensor.detach()
    fused_imgs_captions = fused_imgs_captions.detach()
    #fused_imgs_captions.requires_grad_(False)
    #print("embedding", fused_imgs_captions.shape)
    labels = torch.tensor(labels, dtype=torch.long) # ottengo un unico tensore che contiene tutte le labels (n_embeddings = 198)
    #print("labels", labels.shape)

    hybrid_dataset = TensorDataset(fused_imgs_captions, labels)
    data_loader = DataLoader(hybrid_dataset, batch_size=5, shuffle=True)

    return data_loader
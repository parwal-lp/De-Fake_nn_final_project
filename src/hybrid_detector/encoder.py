import pandas as pd
import torch
import os
import clip
from PIL import Image
from torchvision import transforms

# these instructions are needed to:
# - encode an input image using CLIP image encoder
# - encode its label using CLIP text encoder
# - concatenate these two embeddings into a single item (that becomes the input for the training of our classifier)

clip_dir = "../CLIP/"
os.chdir(clip_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def encode_images_and_captions(captions_file, real_img_dir, fake_img_dir):

    os.chdir("../De-Fake_nn_final_project")
    print(os.getcwd())

    img_dirs = [real_img_dir, fake_img_dir]
    df = pd.read_csv(captions_file)

    encoded_images = []
    encoded_captions = []
    labels = []

    #create encodings for images and texts
    for dir in img_dirs:
        for img_name in os.listdir(dir): #scorro prima le real e poi le fake
            print(dir, img_name)
            img_path = os.path.join(dir, img_name)
            img = Image.open(img_path)

            #tensor_img = transforms.ToTensor()(img).to(device)
            tensor_img = preprocess(img).unsqueeze(0).to(device)
            encoded_img = model.encode_image(tensor_img)

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
            tensor_caption = torch.cat([clip.tokenize(caption)]).to(device)
            encoded_caption = model.encode_text(tensor_caption)

            encoded_images.append(encoded_img)
            encoded_captions.append(encoded_caption)
            labels.append(img_class)

    print(labels)
    return encoded_images, encoded_captions, labels


def fuse_embeddings(encoded_images, encoded_labels):
    fused_embeddings = []
    for img, lab in zip(encoded_images, encoded_labels):
        img_lab = torch.cat((img, lab), dim=1)
        fused_embeddings.append(img_lab)
    return fused_embeddings


#inizio con LD perche e gratis e facile da rigenerare se rompo la directory
#captions_file = "data/hybrid_detector_data/val_LD/mscoco_captions.csv"
#real_img_dir = "data/hybrid_detector_data/val_LD/class_1"
#fake_img_dir = "data/hybrid_detector_data/val_LD/class_0"

#imgs, captions, labels = encode_images_and_captions(captions_file, real_img_dir, fake_img_dir)
#fused_imgs_captions = fuse_embeddings(imgs, captions)

#print(fused_imgs_captions)
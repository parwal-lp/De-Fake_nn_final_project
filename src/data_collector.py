from pycocotools.coco import COCO
import random
import urllib
import os

import deeplake
import PIL
from PIL import Image


def fetchImagesFromMSCOCO(save_img_dir, save_caption_dir, n_instances):
    # Provide the path to the annotations file
    # dataDir = 'data'
    dataType = 'train2017'  # Select the appropriate dataset subset
    # instancesFile = f'{dataDir}/MSCOCO/annotations/annotations_trainval2017/instances_{dataType}.json'
    # captionsFile = f'{dataDir}/MSCOCO/annotations/annotations_trainval2017/captions_{dataType}.json'

    instancesFile = f'annotations_trainval2017/annotations/instances_{dataType}.json'
    captionsFile = f'annotations_trainval2017/annotations/captions_{dataType}.json'


    if not os.path.isdir(save_img_dir):
        os.makedirs(save_img_dir)

    if not os.path.isdir(save_caption_dir):
        os.makedirs(save_caption_dir)

    # Initialize COCO API
    cocoImages = COCO(instancesFile)
    cocoCaptions = COCO(captionsFile)

    # Get all image IDs
    all_image_ids = cocoImages.getImgIds()

    # Select N random image IDs
    random_image_ids = random.sample(all_image_ids, n_instances)

    # prepara il file csv in cui scrivere i dati delle immagini recuperate
    with open(f'{save_caption_dir}/mscoco_captions.csv', 'a') as f:
        intestazione_tabella = "img_id,ann_id,caption"
        print(intestazione_tabella, file=f)

    # Download and save the selected images
    for img_id in random_image_ids:
        img_info = cocoImages.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        #salva ogni immagine in un file che ha il nome del suo id
        img_file_name = os.path.join(save_img_dir, f"{img_id}.jpg")
        urllib.request.urlretrieve(img_url, img_file_name)

        #recupero le caption relative a ogni immagine
        annotation_id = cocoCaptions.getAnnIds(imgIds=img_id)
        annotations = cocoCaptions.loadAnns(annotation_id)
        first_caption = annotations[0]['caption'] #salvo solo la prima caption di ogni immagine perche me ne serve solo una
        with open(f'{save_caption_dir}/mscoco_captions.csv', 'a') as f:
            print(f'{img_id},{annotations[0]["id"]},"{first_caption}"', file=f) #scrivo le caption nel file csv preparato prima


# Set the directory to save the images and the captions
# save_img_dir = f'{dataDir}/MSCOCO/images'
# save_caption_dir = f'{dataDir}/MSCOCO'
# n_instances = 50

# fetchImagesFromMSCOCO(save_img_dir=save_img_dir, save_caption_dir=save_caption_dir, n_instances=n_instances)


# da qui test flikr
'''
def fetchImagesFromFlikr30k(save_img_dir, save_caption_dir, n_instances):
    dataset = deeplake.load('hub://activeloop/flickr30k')
    print("dataset loaded!")
    image_data = dataset.tensors['image'][0].numpy().tobytes()
    print(image_data)



    img_file_name = os.path.join(save_img_dir, "test.jpg")

    with open(img_file_name, "wb") as image_file:
        image_file.write(image_data)

    # creating a image object (main image) 
    img_file = Image.open(image_data) 
    # save a image using extension
    img_file.save(img_file_name)

    images = dataset.tensors['image'].numpy(aslist=True)
    captions = dataset.tensors['caption_0'].numpy()

    random_indices = random.sample(range(len(images)), n_instances)
    print("Selected the following random indices:", random_indices)

    random_images = [images[i] for i in random_indices]
    random_captions = [captions[i] for i in random_indices]

    # prepara il file csv in cui scrivere i dati delle immagini recuperate
    with open(f'{save_caption_dir}/mscoco_captions.csv', 'a') as f:
        intestazione_tabella = "img_id,caption"
        print(intestazione_tabella, file=f)
    print("generato file captions")
    # Download and save the selected images
    for img, caption in zip(random_images, random_captions):
        print(img, caption)
        #salva ogni immagine in un file che ha il nome del suo id
        img_id = img['id']
        img_file_name = os.path.join(save_img_dir, f"{img_id}.jpg")
        # creating a image object (main image) 
        img_file = Image.open(img) 
  
        # save a image using extension
        img_file = img_file.save(img_file_name)

        #recupero le caption relative a ogni immagine
        with open(f'{save_caption_dir}/mscoco_captions.csv', 'a') as f:
            print(f'{img_id},"{caption}"', file=f) #scrivo le caption nel file csv preparato prima

fetchImagesFromFlikr30k("data/test_flikr", "data/test_flikr", 10)
'''
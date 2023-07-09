import os
import shutil

#this script generates a dataset for training or evaluating the image-only detector
#structure of the dataset generated
# train/ (or val_LD/ or val_GLIDE/ etc)
#     ├── class_0/
#     │   ├── ...
#     │   └── all the fake images
#     ├── class_1/
#     │   ├── ...
#     │   └── all the real images

#notare che lo script si occupa anche di identificare tutte le fake images che non si sono generate correttamente.
#Queste vengono quindi escluse, e non viene riportata ne la fake invalida, ne la corrispondente real image

def formatIntoDataset(real_imgs_path, fake_imgs_path, dest_dataset_dir):
    invalid_images = []
    #itera tutte le real images
    for file_name in os.listdir(fake_imgs_path):
        img_path = os.path.join(fake_imgs_path, file_name)
        
        # Check if the file is an image (you can modify this condition based on your image file extensions)
        if os.path.isfile(img_path):
            # Check if the excluded string is not present in the file name
            if "invalid_prompt" in file_name:
                invalid_images.append(file_name[5:-19]+".jpg")
            else:
                dest_path = os.path.join(f"{dest_dataset_dir}/class_0", file_name)
                shutil.copy2(img_path, dest_path)
    print(invalid_images)

    for file_name in os.listdir(real_imgs_path):
        img_path = os.path.join(real_imgs_path, file_name)
        
        # Check if the file is an image (you can modify this condition based on your image file extensions)
        if os.path.isfile(img_path):
            # Check if the excluded string is not present in the file name
            if file_name not in invalid_images:
                dest_path = os.path.join(f"{dest_dataset_dir}/class_1", file_name)
                shutil.copy2(img_path, dest_path)

path_real_images = "data/MSCOCO/images"
path_fake_images = "data/SD+MSCOCO/images"
dest_dataset_dir = "data/imageonly_detector_data/train"

formatIntoDataset(path_real_images, path_fake_images, dest_dataset_dir)
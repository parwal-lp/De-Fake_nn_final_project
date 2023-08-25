import os
import shutil


def format_dataset_multiclass(SD_dir, LD_dir, GLIDE_dir, dest_dataset_dir):

    if not os.path.isdir(f"{dest_dataset_dir}/class_LD"): os.makedirs(f"{dest_dataset_dir}/class_LD")
    if not os.path.isdir(f"{dest_dataset_dir}/class_SD"): os.makedirs(f"{dest_dataset_dir}/class_SD")
    if not os.path.isdir(f"{dest_dataset_dir}/class_GLIDE"): os.makedirs(f"{dest_dataset_dir}/class_GLIDE")
    invalid_images = []

    print("sposto le immagini LD")
    for file_name in os.listdir(LD_dir):
            img_path = os.path.join(LD_dir, file_name)
            # Check if the file is an image (you can modify this condition based on your image file extensions)
            if os.path.isfile(img_path):
                dest_path = os.path.join(f"{dest_dataset_dir}/class_LD", file_name)
                shutil.move(img_path, dest_path)

    print("sposto le immagini SD")
    for file_name in os.listdir(SD_dir):
            img_path = os.path.join(SD_dir, file_name)
            # Check if the file is an image (you can modify this condition based on your image file extensions)
            if os.path.isfile(img_path):
                # Check if the excluded string is not present in the file name
                if "invalid_prompt" in file_name:
                    invalid_images.append(file_name[5:-19]+".jpg")
                else:
                    dest_path = os.path.join(f"{dest_dataset_dir}/class_SD", file_name)
                    shutil.move(img_path, dest_path)
    print(invalid_images)

    print("sposto le immagini GLIDE")
    for file_name in os.listdir(GLIDE_dir):
            img_path = os.path.join(GLIDE_dir, file_name)
            # Check if the file is an image (you can modify this condition based on your image file extensions)
            if os.path.isfile(img_path):
                dest_path = os.path.join(f"{dest_dataset_dir}/class_GLIDE", file_name)
                shutil.move(img_path, dest_path)


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

def format_dataset_binaryclass(fake_imgs_path, dest_dataset_dir):
    if not os.path.isdir(f"{dest_dataset_dir}/class_0"): os.makedirs(f"{dest_dataset_dir}/class_0")

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
                shutil.move(img_path, dest_path)
    print(f"invalid images detected and not moved into the new dataset: {invalid_images}")


def formatIntoTrainTest(real_imgs_path, fake_imgs_path, dest_dataset_dir):
    if not os.path.isdir(f"{dest_dataset_dir}/train/class_1"): os.makedirs(f"{dest_dataset_dir}/train/class_1")
    if not os.path.isdir(f"{dest_dataset_dir}/val/class_1"): os.makedirs(f"{dest_dataset_dir}/val/class_1")
    if not os.path.isdir(f"{dest_dataset_dir}/train/class_0"): os.makedirs(f"{dest_dataset_dir}/train/class_0")
    if not os.path.isdir(f"{dest_dataset_dir}/val/class_0"): os.makedirs(f"{dest_dataset_dir}/val/class_0")

    invalid_images = []
    #itera tutte le real images
    print("iteration on fake images")
    index = 1
    for file_name in os.listdir(fake_imgs_path):
        if index==50:
            print("passo a val")
        img_path = os.path.join(fake_imgs_path, file_name)
        
        # Check if the file is an image (you can modify this condition based on your image file extensions)
        if os.path.isfile(img_path):
            # Check if the excluded string is not present in the file name
            if "invalid_prompt" in file_name:
                invalid_images.append(file_name[5:-19]+".jpg")
            else:
                if index <= 50:
                    dest_path = os.path.join(f"{dest_dataset_dir}/train/class_0", file_name)
                else:
                    dest_path = os.path.join(f"{dest_dataset_dir}/val/class_0", file_name)
                shutil.move(img_path, dest_path)
                index += 1
    print(f"invalid images detected and not moved into the new dataset: {invalid_images}")

    print("iteration on real images")
    index = 1
    for file_name in os.listdir(real_imgs_path):
        if index==50:
            print("passo a val")
        img_path = os.path.join(real_imgs_path, file_name)
        
        # Check if the file is an image (you can modify this condition based on your image file extensions)
        if os.path.isfile(img_path):
            # Check if the excluded string is not present in the file name
            if file_name not in invalid_images:
                if index <= 50:
                    dest_path = os.path.join(f"{dest_dataset_dir}/train/class_1", file_name)
                else:
                    dest_path = os.path.join(f"{dest_dataset_dir}/val/class_1", file_name)
                shutil.move(img_path, dest_path)
                index += 1
            else:
                # if the real image is associated to an invalid prompt, delete the image
                os.remove(img_path)


#path_real_images = "data/MSCOCO/images"
#path_fake_images = "data/SD+MSCOCO/images"
#dest_dataset_dir = "data/imageonly_detector_data/train"

#formatIntoDataset(path_real_images, path_fake_images, dest_dataset_dir)

#formatIntoTrainTest("data/MSCOCO_for_SD/images", "data/SD+MSCOCO/images", "data/imageonly_detector_data")
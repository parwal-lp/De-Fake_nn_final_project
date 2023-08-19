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
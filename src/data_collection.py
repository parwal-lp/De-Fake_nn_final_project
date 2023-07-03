from pycocotools.coco import COCO
import random
import urllib
import os

# Provide the path to the annotations file
dataDir = 'data'
dataType = 'train2017'  # Select the appropriate dataset subset
annFile = f'{dataDir}/annotations/instances_{dataType}.json'

# Initialize COCO API
coco = COCO(annFile)

# Get all image IDs
all_image_ids = coco.getImgIds()

# Select 200 random image IDs
random_image_ids = random.sample(all_image_ids, 500)

# Set the directory to save the images
save_dir = 'data/images'

# Download and save the selected images
for img_id in random_image_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']
    img_file_name = os.path.join(save_dir, f"{img_id}.jpg")
    urllib.request.urlretrieve(img_url, img_file_name)

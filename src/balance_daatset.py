import os
import shutil
from collections import defaultdict
import random

src_dir = 'dataset\\FER-2013\\train'
dst_dir = 'dataset\\FER-2013\\balanced_train'
num_images_per_class = 1000  # adjust as needed

os.makedirs(dst_dir, exist_ok=True)

for class_name in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)
    selected_images = images[:num_images_per_class]

    new_class_path = os.path.join(dst_dir, class_name)
    os.makedirs(new_class_path, exist_ok=True)

    for img in selected_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(new_class_path, img))

import os
import cv2
import random
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from random import shuffle
from itertools import cycle, islice

from random import choice

from mltu.annotations.images import CVImage

# Define or import your augmentors
# These should be callable: aug(image) → augmented image
from src.train.augmentors import RandomRotate, RandomHorizontalScale, RandomHorizontalShear

from src.train.preprocessors import ImageReader

# Instantiate augmentors
augmentors = [
    RandomRotate(random_chance=1, angle=10),
    RandomHorizontalScale(random_chance=1),
    RandomHorizontalShear(random_chance=1, max_shear_factor=0.3),
]

# Paths

output_dir = "data/hcs_augmented_combination_dataset"
os.makedirs(output_dir, exist_ok=True)


train_dataset = np.load("data/trainset_hcs.npy")
train_dataset = [(name, label) for name, label in train_dataset]

image_reader = ImageReader(CVImage)


k = 0

train_list = []

for img_path, label in train_dataset:

    image, _ = image_reader(img_path, label)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    print(f"Augmenting {k} image {img_name}")

    orig_image = image.numpy() # np array

    

    for j in range(10):

        aug_type_string = []
        full_augmented_image = image

        n = random.randint(1, 3)
        aug_list = random.sample(augmentors, n)
        random.shuffle(aug_list)

        for i, aug in enumerate(aug_list):  # generate 10 augmented versions

            full_augmented_image, _ = aug(full_augmented_image, label)

            aug_name = type(aug).__name__

            aug_type_string.append(aug_name)
        
        final_name = "_".join(aug_type_string)

        aug_filename = f"{img_name}_{j}_{final_name}.png"
        aug_path = os.path.join(output_dir, aug_filename)

        aug_np = full_augmented_image.numpy()
        
        cv2.imwrite(aug_path, aug_np)

        full_augmented_image.update(orig_image)

        # Optionally, write new label entry
        with open(os.path.join(output_dir, "labels.txt"), "a") as out_f:
            out_f.write(f"{aug_path},{label}\n")

        tup = (aug_path,label)

        train_list.append(tup)

    k =  k + 1

print("Saving..")

np.save("data/trainset_hcs_augmented_combined.npy", train_list)

print("Done!")
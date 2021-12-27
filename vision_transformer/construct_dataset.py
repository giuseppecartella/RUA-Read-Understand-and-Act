import os
import random
import cv2
import matplotlib.pyplot as plt
from random_augmentation import RandomImgOperator

def get_dataset_path(dataset_name):
    return os.path.join(os.path.dirname(__file__), dataset_name)

ade_20k = os.path.join(os.path.dirname(__file__), 'ADE20K_2021_17_01', 'images','ADE', 'training')
resized_dataset = 'resized_dataset'
envs = ['home_or_hotel', 'shopping_and_dining', 'work_place', 'urban', 'cultural']
paths = [os.path.join(ade_20k,env) for env in envs]
final_dataset = 'dataset'


#PARAMETERS
back_img_size = (640, 480)
default_sign_size = (200, 100)
num_classes = 9
imgs_per_variant = {"0": 10000, "1": 1400, "2": 1400, "3": 1650, "4": 1650, "5": 1650, "6": 1650, "7": 5000, "8": 5000}

dataset_path = get_dataset_path(final_dataset)
if not os.path.isdir(dataset_path):
    #create directory for the dataset and all subdirectories.
    os.makedirs(dataset_path)
    for i in range(num_classes):
        os.makedirs(os.path.join(dataset_path, str(i)))

resized_dataset_path = get_dataset_path(resized_dataset)
if not os.path.isdir(resized_dataset_path):
    #create directory for the dataset and all subdirectories.
    os.makedirs(resized_dataset_path)

    for path in paths:
        for root, dirs, files in os.walk(path):
            for filename in files:
                if 'frame' in filename:
                    continue
                if filename.endswith('.jpg'):
                    img = cv2.imread(os.path.join(root, filename))
                    img = cv2.resize(img, back_img_size)
                    cv2.imwrite(os.path.join(get_dataset_path(resized_dataset), filename), img)


#construct dataset
random_img_augmenter = RandomImgOperator()
background_imgs = os.listdir(resized_dataset_path)
dir_list = os.listdir(dataset_path)
signs_path = get_dataset_path('SIGNS')

for dir in dir_list:
    print('Constructing images for class {}...'.format(dir))

    idx = 0
    if dir == "0":
        selected_imgs = random.sample(background_imgs, imgs_per_variant[dir])
        random_backgrounds = random.sample(background_imgs, imgs_per_variant[dir])
        for background_file in random_backgrounds:
            random_background_path = os.path.join(resized_dataset_path, background_file)
            background = cv2.imread(random_background_path)
            result = random_img_augmenter.apply_random_operations(background, None, only_background=True)
            cv2.imwrite(os.path.join(dataset_path, dir, str(idx) + '.jpg'), result)
            idx += 1
    else:
        class_sign_path = os.path.join(signs_path, dir)
        
        for file in os.listdir(class_sign_path):
            file_name = os.path.join(class_sign_path, file)
            sign = cv2.imread(file_name)
            sign = cv2.resize(sign, default_sign_size)

            random_backgrounds = random.sample(background_imgs, imgs_per_variant[dir])
            
            for background_file in random_backgrounds:
                random_background_path = os.path.join(resized_dataset_path, background_file)
                background = cv2.imread(random_background_path)
                result = random_img_augmenter.apply_random_operations(background, sign.copy())
                cv2.imwrite(os.path.join(dataset_path, dir, str(idx) + '.jpg'), result)
                idx += 1
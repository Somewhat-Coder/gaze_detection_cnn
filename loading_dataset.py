import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random

random.seed(42)

DATASET_DIR = 'Columbia Gaze Data Set'

def load_images_and_labels(dataset_dir):
    images = []
    labels = []

    for subject_dir in os.listdir(dataset_dir):
        subject_path = os.path.join(dataset_dir, subject_dir)
        
        if os.path.isdir(subject_path):
            for img_name in os.listdir(subject_path):
                img_path = os.path.join(subject_path, img_name)
                
                image = cv2.imread(img_path)

                if image is None:
                    continue 
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                image = cv2.resize(image, (224, 224))

                images.append(image)
                
                label = get_label_from_filename(img_name)
                labels.append(label)

    return np.array(images), np.array(labels)


def get_label_from_filename(filename):

    parts = filename.split('_')

    parts[-1] = parts[-1].split('.')[0]

    vertical_angle = float(parts[-2][:-1])
    horizontal_angle = float(parts[-1][:-1])

    return np.array([horizontal_angle, vertical_angle])


images, labels = load_images_and_labels(DATASET_DIR)

np.save('images.npy', images) 
np.save('labels.npy', labels)



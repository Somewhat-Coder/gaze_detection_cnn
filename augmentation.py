import numpy as np
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


images = np.load('images.npy')
labels = np.load('labels.npy')

def add_gaussian_noise(image):
    mean = 0
    stddev = 0.05 
    gauss = np.random.normal(mean, stddev, image.shape)
    noisy_image = np.clip(image + gauss, 0, 1)
    return noisy_image

data_gen = ImageDataGenerator(
    brightness_range=[0.8, 1.2], 
    zoom_range=0.1,           
    horizontal_flip=True,       
    rescale=1./255              
)

def augment_images(images, target_count=50000):
    augmented_images = []
    augmented_labels = [] 
    count = 0
    batch_size = 32

    for batch, labels_batch in data_gen.flow(images, labels, batch_size=batch_size, shuffle=True):

        noisy_batch = np.array([add_gaussian_noise(img) for img in batch])
        
        augmented_images.append(noisy_batch)
        augmented_labels.append(labels_batch)
        
        count += len(noisy_batch)
        if count >= target_count:
            break
    
    return np.vstack(augmented_images), np.vstack(augmented_labels)

augmented_images, augmented_labels = augment_images(images, target_count=4000)

np.save('augmented_images.npy', augmented_images)
np.save('augmented_labels.npy', augmented_labels)

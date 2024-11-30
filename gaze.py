import tensorflow as tf
from tensorflow.keras import layers, Model, applications

def create_face_detection_model(input_shape=(224, 224, 3)):
    """Create a face detection model using a pre-trained MobileNetV2."""
    base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    output = layers.Dense(4, activation='linear', name='face_bbox')(x)  # [x, y, width, height]
    
    model = Model(inputs=base_model.input, outputs=output, name='face_detection')
    return model

def create_eye_detection_model(input_shape=(96, 96, 3)):
    """Create an eye detection model."""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(8, activation='linear', name='eyes_bbox')(x)  # [x1, y1, w1, h1, x2, y2, w2, h2] for both eyes
    
    model = Model(inputs=inputs, outputs=output, name='eye_detection')
    return model

def create_gaze_direction_model(input_shape=(32, 32, 3)):
    """Create a gaze direction classification model."""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(3, activation='softmax', name='gaze_direction')(x)  # [left, straight, right]
    
    model = Model(inputs=inputs, outputs=output, name='gaze_direction')
    return model

def crop_and_resize(image, bbox, target_size):
    """Crop the image using the bounding box and resize to target size."""
    x, y, w, h = [bbox[i] for i in range(4)]
    x1 = tf.clip_by_value(x, 0, 1)
    y1 = tf.clip_by_value(y, 0, 1)
    x2 = tf.clip_by_value(x+w, 0, 1)
    y2 = tf.clip_by_value(y+h, 0, 1)
    cropped = tf.image.crop_and_resize(image, [[y1, x1, y2, x2]], [0], target_size)
    return cropped[0]

def create_cascaded_model(input_shape=(224, 224, 3)):
    """Create a cascaded model for face detection, eye detection, and gaze direction classification."""
    face_model = create_face_detection_model(input_shape)
    eye_model = create_eye_detection_model((96, 96, 3))
    gaze_model = create_gaze_direction_model((32, 32, 3))
    
    # Freeze the weights of pre-trained models if necessary
    face_model.trainable = False
    eye_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    
    # Face detection
    face_bbox = face_model(inputs)
    
    # Crop and resize face
    face = layers.Lambda(lambda x: crop_and_resize(x[0], x[1], (96, 96)))([inputs, face_bbox])
    
    # Eye detection
    eyes_bbox = eye_model(face)
    
    # Crop and resize eyes
    left_eye = layers.Lambda(lambda x: crop_and_resize(x[0], x[1][:4], (32, 32)))([face, eyes_bbox])
    right_eye = layers.Lambda(lambda x: crop_and_resize(x[0], x[1][4:], (32, 32)))([face, eyes_bbox])
    
    # Gaze direction for each eye
    left_gaze = gaze_model(left_eye)
    right_gaze = gaze_model(right_eye)
    
    # Average the gaze directions
    avg_gaze = layers.Average()([left_gaze, right_gaze])
    
    cascaded_model = Model(inputs=inputs, outputs=[face_bbox, eyes_bbox, avg_gaze])
    return cascaded_model

# Create the cascaded model
model = create_cascaded_model()

# Compile the model
model.compile(optimizer='adam',
              loss={'face_detection': 'mse', 
                    'eye_detection': 'mse', 
                    'gaze_direction': 'categorical_crossentropy'},
              loss_weights={'face_detection': 1.0, 
                            'eye_detection': 1.0, 
                            'gaze_direction': 1.0})

# Print model summary
model.summary()

# Example of how to use the model for inference
def predict_gaze(image):
    # Preprocess the image (resize to 224x224 and normalize)
    preprocessed_image = tf.image.resize(image, (224, 224))
    preprocessed_image = applications.mobilenet_v2.preprocess_input(preprocessed_image)
    
    # Make prediction
    face_bbox, eyes_bbox, gaze_direction = model.predict(tf.expand_dims(preprocessed_image, 0))
    
    # Convert gaze direction to class
    gaze_class = ['Left', 'Straight', 'Right'][tf.argmax(gaze_direction[0])]
    
    return face_bbox[0], eyes_bbox[0], gaze_class

# You would typically load pre-trained weights here
# model.load_weights('path_to_pretrained_weights.h5')

# Example usage
# import numpy as np
# from PIL import Image
# image = np.array(Image.open('path_to_your_image.jpg'))
# face_bbox, eyes_bbox, gaze_class = predict_gaze(image)
# print(f"Face bounding box: {face_bbox}")
# print(f"Eyes bounding boxes: {eyes_bbox}")
# print(f"Gaze direction: {gaze_class}")
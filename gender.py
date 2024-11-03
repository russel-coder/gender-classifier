import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import sys
import io

# Set paths to your dataset
train_dir = r'C:\Users\HP\Desktop\gender classifier\train'
test_dir = r'C:\Users\HP\Desktop\gender classifier\test'

# Redirect stdout to handle encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Preprocessing
train_datagen = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)
"""
# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
"""

""""
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)
"""

model = load_model(r'C:\Users\HP\Desktop\gender classifier\gender_model.h5')

def classify_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch axis

    # Normalize the image
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    
    # Determine class and confidence
    if confidence > 0.5:

        class_label = 'Male'
        confidence_percentage = confidence * 100
    else:
        class_label = 'Female'
        confidence_percentage = (1 - confidence) * 100

    return class_label, confidence_percentage

# Example usage of classify_image function
image_path = r'C:\Users\HP\Desktop\gender classifier\train\Female Faces\vvv.jpg'  # Update with your image path
result_label, confidence = classify_image(image_path)
print(f'The image is classified as: {result_label} with a confidence of {confidence:.2f}%')

# Function to plot prediction confidence
def plot_prediction_confidence(label, confidence):
    plt.figure(figsize=(10, 6))
    
    # Determine confidence values for plotting
    if label == 'Male':
        values = [confidence, 100 - confidence]
    else:
        values = [100 - confidence, confidence]

    categories = ['Male', 'Female']
    
    plt.bar(categories, values, color=['green', 'blue'])
    plt.title(f'Prediction Confidence: {label}')
    plt.xlabel('Class')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 100)
    
    # Annotate bars with their values
    for i, v in enumerate(values):
        plt.text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom')

    plt.show()

# Plot the prediction confidence
plot_prediction_confidence(result_label, confidence)

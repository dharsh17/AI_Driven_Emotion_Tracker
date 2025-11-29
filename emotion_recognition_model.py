import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Hyperparameters
img_width, img_height = 48, 48
batch_size = 32
epochs = 20

# Data augmentation and loading
train_data_gen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=20, horizontal_flip=True)
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_data_gen = ImageDataGenerator(rescale=1.0/255.0)
validation_generator = validation_data_gen.flow_from_directory(
    'data/validation',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the model
model.save('emotion_recognition_model.h5')

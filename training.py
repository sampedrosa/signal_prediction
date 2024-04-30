import os
import pickle
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, concatenate
from keras.utils import load_img, img_to_array, to_categorical
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence

# Get the Inputs (image.jpg, landmarks.pkl) for Training a Neural Network Model with Mixed Inputs (CNN and FNN)
###############################################################################################################

LABELS = ['C', 'Segunda', 'Casa', 'Obrigado', 'Rir', 'Aviao', 'Vazio']  # Signals for Training
IMG_SHAPE = (480, 640, 3)
LM_SHAPE = (3, 21, 3)
BS = 8    # Batch-Size
EPS = 20  # Epochs

# CustomDataGenerator receiving an image and landmarks coordinates as inputs for classification (labels)
class CustomDataGenerator(Sequence):
    def __init__(self, path):
        self.img_dir = path + '/images/'     # Path for image directory
        self.lms_dir = path + '/landmarks/'  # Path for landmarks directory
        self.batch_size = BS
        self.image_size = IMG_SHAPE[:-1]
        self.labels = sorted(LABELS)
        self.indexes = np.arange(len(self.labels))
        self.file_names = self.get_file_names()

    def get_file_names(self):
        file_names = []
        for class_name in self.labels:
            image_files = os.listdir(os.path.join(self.img_dir, class_name))
            for image_file in image_files:
                if image_file.endswith('.jpg'):
                    landmarks_file = image_file[:-4] + '.pkl'
                    file_names.append((class_name, image_file, landmarks_file))
        return file_names

    def __len__(self):
        return int(np.ceil(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_names[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_files)
        return X, y

    def __data_generation(self, batch_files):
        X_img, X_lms, y = [], [], []

        for label_name, img_file, lms_file in batch_files:
            img_path = os.path.join(self.img_dir, label_name, img_file)
            lms_path = os.path.join(self.lms_dir, label_name, lms_file)

            # Load Image
            image = load_img(img_path, target_size=self.image_size)
            image = img_to_array(image) / 255.0  # Normalize
            X_img.append(image)

            # Load Landmarks
            with open(lms_path, 'rb') as f:
                lms_data = pickle.load(f)
            X_lms.append(lms_data)

            # Assign class label
            y.append([self.labels.index(label_name)])

        X_img = np.array(X_img)  # Image Input
        X_lms = np.array(X_lms)  # Landmark Coordinates Input
        y = to_categorical(np.array(y), num_classes=len(self.labels))  # Classification baes on files
        return [X_img, X_lms], y

# Creating Generators for Training and Validation
train_generator = CustomDataGenerator('./data/train')
val_generator = CustomDataGenerator('./data/validation')

# Image Input Layers (CNN - Convolutional Neural Network)
input_img = Input(shape=IMG_SHAPE)
img = Conv2D(BS*4, kernel_size=(3, 3), activation='relu')(input_img)
img = MaxPooling2D(pool_size=(2, 2))(img)
img = Dropout(0.1)(img)
img = Conv2D(BS*8, kernel_size=(3, 3), activation='relu')(img)
img = MaxPooling2D(pool_size=(2, 2))(img)
img = Dropout(0.1)(img)
img = Flatten()(img)

# Landmarks Coordinates Input Layers (FNN - Feed-Forward Neural Networks)
input_lms = Input(shape=LM_SHAPE)
lms = Dense(BS*8, activation='relu')(input_lms)
lms = Dropout(0.1)(lms)
lms = Dense(BS*4, activation='relu')(lms)
lms = Dropout(0.1)(lms)
lms = Reshape((-1,))(lms)

# Merging Both Inputs for Output Layers
merged = concatenate([img, lms])
merged = Dense(BS*4, activation='relu')(merged)
output = Dense(len(LABELS), activation='softmax')(merged)

# Creating and Training Model
model = Model(inputs=[input_img, input_lms], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cb = ModelCheckpoint('./models/signals_model.h5', save_best_only=True)
model.fit(train_generator, validation_data=val_generator, epochs=EPS, callbacks=[cb])
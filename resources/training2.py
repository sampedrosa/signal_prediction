import cv2
import os
import sys
import pickle
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Concatenate, concatenate, GlobalAveragePooling2D, Input, Dot, Add
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array, to_categorical
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from keras.models import load_model
from keras.callbacks import TensorBoard

CROP = (200, 200, 3)
FRAME = (480, 640, 3)
LABELS = ['C', 'Segunda', 'Casa', 'Obrigado', 'Rir', 'Aviao', 'Vazio']
bs = 8
ep = 10

#train_datagen_h1 = ImageDataGenerator(rescale=1./255)
#train_h1 = train_datagen_h1.flow_from_directory('./data/train/hand1', target_size=(CROP[0], CROP[1]), batch_size=bs, class_mode='categorical')
#train_datagen_h2 = ImageDataGenerator(rescale=1./255)
#train_h2 = train_datagen_h2.flow_from_directory('./data/train/hand2', target_size=(CROP[0], CROP[1]), batch_size=bs, class_mode='categorical')
#train_datagen_hlm1 = ImageDataGenerator(rescale=1./255)
#train_hlm1 = train_datagen_hlm1.flow_from_directory('./data/train/hand1 landmarks', target_size=(CROP[0], CROP[1]), batch_size=bs, class_mode='categorical')
#train_datagen_hlm2 = ImageDataGenerator(rescale=1./255)
#train_hlm2 = train_datagen_hlm2.flow_from_directory('./data/train/hand2 landmarks', target_size=(CROP[0], CROP[1]), batch_size=bs, class_mode='categorical')
#val_datagen_h1 = ImageDataGenerator(rescale=1./255)
#val_h1 = val_datagen_h1.flow_from_directory('./data/validation/hand1', target_size=(CROP[0], CROP[1]), batch_size=bs, class_mode='categorical')
#val_datagen_h2 = ImageDataGenerator(rescale=1./255)
#val_h2 = val_datagen_h2.flow_from_directory('./data/validation/hand2', target_size=(CROP[0], CROP[1]), batch_size=bs, class_mode='categorical')
#val_datagen_hlm1 = ImageDataGenerator(rescale=1./255)
#val_hlm1 = val_datagen_hlm1.flow_from_directory('./data/validation/hand1 landmarks', target_size=(CROP[0], CROP[1]), batch_size=bs, class_mode='categorical')
#val_datagen_hlm2 = ImageDataGenerator(rescale=1./255)
#val_hlm2 = val_datagen_hlm2.flow_from_directory('./data/validation/hand2 landmarks', target_size=(CROP[0], CROP[1]), batch_size=bs, class_mode='categorical')
'''
input_h1 = Input(shape=CROP)
h1 = Conv2D(bs*4, kernel_size=(3, 3), activation='relu')(input_h1)
h1 = MaxPooling2D(pool_size=(2, 2))(h1)
h1 = Dropout(0.2)(h1)
h1 = Conv2D(bs*2, kernel_size=(3, 3), activation='relu')(h1)
h1 = MaxPooling2D(pool_size=(2, 2))(h1)
h1 = Dropout(0.2)(h1)
h1 = Conv2D(bs, kernel_size=(3, 3), activation='relu')(h1)
h1 = MaxPooling2D(pool_size=(2, 2))(h1)
h1 = Dropout(0.1)(h1)
h1 = Flatten()(h1)
h1 = Model(inputs=input_h1, outputs=h1)
'''
'''
input_h2 = Input(shape=CROP)
h2 = Conv2D(bs, kernel_size=(3, 3), activation='relu')(input_h2)
h2 = MaxPooling2D(pool_size=(2, 2))(h2)
h2 = Dropout(0.2)(h2)
h2 = Conv2D(bs * 2, kernel_size=(3, 3), activation='relu')(h2)
h2 = MaxPooling2D(pool_size=(2, 2))(h2)
h2 = Dropout(0.4)(h2)
h2 = Conv2D(bs, kernel_size=(3, 3), activation='relu')(h2)
h2 = MaxPooling2D(pool_size=(2, 2))(h2)
h2 = Dropout(0.1)(h2)
h2 = Flatten()(h2)
h2 = Dense(bs, activation='relu')(h2)
h2 = Model(inputs=input_h2, outputs=h2)
'''
'''
input_hlm1 = Input(shape=CROP)
hlm1 = Conv2D(bs*2, kernel_size=(3, 3), activation='relu')(input_hlm1)
hlm1 = MaxPooling2D(pool_size=(2, 2))(hlm1)
hlm1 = Dropout(0.2)(hlm1)
hlm1 = Conv2D(bs, kernel_size=(3, 3), activation='relu')(hlm1)
hlm1 = MaxPooling2D(pool_size=(2, 2))(hlm1)
hlm1 = Dropout(0.1)(hlm1)
hlm1 = Flatten()(hlm1)
hlm1 = Model(inputs=input_hlm1, outputs=hlm1)
'''
'''
input_hlm2 = Input(shape=CROP)
hlm2 = Conv2D(bs, kernel_size=(3, 3), activation='relu')(input_hlm2)
hlm2 = MaxPooling2D(pool_size=(2, 2))(hlm2)
hlm2 = Dropout(0.1)(hlm2)
hlm2 = Dense(bs * 2, activation='relu')(hlm2)
hlm2 = Dense(bs, activation='relu')(hlm2)
hlm2 = Flatten()(hlm2)
hlm2 = Dense(bs, activation='relu')(hlm2)
hlm2 = Model(inputs=input_hlm2, outputs=hlm2)
'''
class CustomDataGenerator(Sequence):
    def __init__(self, g1, g2, g3):
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3

    def __len__(self):
        return len(self.g1)

    def __getitem__(self, idx):
        batch_x1, batch_y1 = self.g1[idx]
        batch_x2, batch_y2 = self.g2[idx]
        batch_x3, batch_y3 = self.g3[idx]
        return [batch_x1, batch_x2, batch_x3], batch_y1

train_datagen_plm = ImageDataGenerator(rescale=1./255)
train_plm = train_datagen_plm.flow_from_directory('./data/train/images', target_size=(FRAME[0], FRAME[1]), batch_size=bs, class_mode='categorical')
val_datagen_plm = ImageDataGenerator(rescale=1./255)
val_plm = val_datagen_plm.flow_from_directory('./data/validation/images', target_size=(FRAME[0], FRAME[1]), batch_size=bs, class_mode='categorical')

input_plm = Input(shape=FRAME)
plm = Conv2D(bs*8, kernel_size=(3, 3), activation='relu')(input_plm)
plm = MaxPooling2D(pool_size=(2, 2))(plm)
plm = Dropout(0.2)(plm)
plm = Conv2D(bs*4, kernel_size=(3, 3), activation='relu')(plm)
plm = MaxPooling2D(pool_size=(2, 2))(plm)
plm = Dropout(0.2)(plm)
plm = Conv2D(bs, kernel_size=(3, 3), activation='relu')(plm)
plm = MaxPooling2D(pool_size=(2, 2))(plm)
plm = Dropout(0.1)(plm)
plm = Flatten()(plm)
plm = Dense(len(LABELS), activation='softmax')(plm)
plm = Model(inputs=input_plm, outputs=plm)

plm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#combined_train_generator = CustomDataGenerator(train_h1, train_hlm1, train_plm)
#combined_val_generator = CustomDataGenerator(val_h1, val_hlm1, val_plm)

cb = ModelCheckpoint('./models/model2.h5', save_best_only=True)
plm.fit(train_plm, steps_per_epoch=train_plm.samples // bs, validation_data=val_plm, epochs=ep, callbacks=[cb])

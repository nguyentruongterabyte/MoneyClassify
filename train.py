import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import decomposition

raw_folder = "data/"
label_amount = 10


def load_data():

    file = open('pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)
    file.close()

    # pixels = pixels.flatten()
    # labels = labels.flatten()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels


X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print(X_train.shape, X_test.shape)


def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
    # Tao model
    input_ = Input(shape=(128, 128, 3), name='pix.data')
    output_vgg16_conv = model_vgg16_conv(input_)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    # Tuy theo so luong nhan
    x = Dense(label_amount, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input_, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model


vggmodel = get_model()

filepath = "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1, rescale=1./255, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, brightness_range=[0.2,1.5], fill_mode="nearest")

aug_val = ImageDataGenerator(rescale=1./255)

vgghist=vggmodel.fit(aug.flow(X_train, y_train, batch_size=64),
                               epochs=50, # steps_per_epoch=len(X_train)//64,
                               validation_data=aug.flow(X_test,y_test,
                               batch_size=64),
                               callbacks=callbacks_list)

vggmodel.save("vggmodel.h5")


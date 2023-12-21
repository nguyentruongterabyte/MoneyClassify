import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

cap = cv2.VideoCapture(0)

class_name = ['000000', '001000', '002000', '005000', '010000', '020000', '050000', '100000', '200000', '500000']


def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input_ = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input_)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(.5)(x)
    x = Dense(len(class_name), activation='softmax', name='predictions')(x)

    my_model_ = Model(inputs=input_, outputs=x)
    my_model_.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model_


my_model = get_model()
# link data train (file weights)
# https://drive.google.com/file/d/10dMRUR5kiWx3MtSfEbthAaW1ztkJ8Gjo/view?usp=sharing
my_model.load_weights('weights-41-0.94.hdf5')

while True:
    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None, fx=.5, fy=.5)

    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float') * 1./255

    image = np.expand_dims(image, axis=0)

    predict = my_model.predict(image)
    print('This picture is: ', class_name[np.argmax(predict[0])], predict[0])
    print(np.max(predict[0], axis=0))

    if np.max(predict) >= 0.6 and np.argmax(predict[0]) != 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

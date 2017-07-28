import json
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1]

class Vgg16:
    def __init__(self):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create()
        self.get_classes()

    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH + fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def create(self):
        self.model = Sequential()
        self.model.add(Lambda(vgg_preprocess,
                input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        self.model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        self.model.add(Dense(1000, activation='softmax'))

    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D(padding=(1, 1)))
            model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))

    def get_batches(self, dir, gen=ImageDataGenerator(), batch_size=32):
        return gen.flow_from_directory(dir, batch_size=batch_size)

    def finetune(self):
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(2, activation='softmax'))

    def fit(self, batches, steps, epochs=1, validation_batches=None, validation_steps=None):
        return self.model.fit_generator(batches, steps, epochs=epochs,
                validation_data=validation_batches,
                validation_steps=validation_steps)

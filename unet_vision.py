import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
import cv2 as cv
import random

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import parameters


class unet_vision:
    def __init__(self):
        self.h = parameters.get_config()

        self.model = self.unet_model()
        self.model.load_weights(self.h.save_model_folder + self.h.save_model_name)

    def conv_block(self, inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
        # Convolutional block with LeakyReLU activation
        conv1 = layers.Conv2D(n_filters, 3, padding='same')(inputs)
        conv1 = layers.LeakyReLU(alpha=0.1)(conv1)

        if batch_norm:
            conv1 = layers.BatchNormalization(axis=-1)(conv1)
        conv2 = layers.Conv2D(n_filters, 3, padding='same')(conv1)
        conv2 = layers.LeakyReLU(alpha=0.1)(conv2)

        if batch_norm:
            conv2 = layers.BatchNormalization(axis=-1)(conv2)

        if dropout_prob > 0:
            conv2 = layers.Dropout(dropout_prob)(conv2)
        return conv2

    def encoder_block(self, inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
        # Encoder block with convolutional block and max pooling
        skip_connection = self.conv_block(inputs, n_filters, batch_norm, dropout_prob)
        next_layer = layers.MaxPooling2D((2, 2))(skip_connection)
        return next_layer, skip_connection

    def decoder_block(self, expansive_input, skip_connection, n_filters, batch_norm=False, dropout_prob=0):
        # Decoder block with transpose convolution and concatenation
        up = layers.Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
        merge = layers.concatenate([up, skip_connection], axis=-1)

        conv = self.conv_block(merge, n_filters, batch_norm, dropout_prob)

        return conv

    def unet_model(self):
        # U-Net model definition
        inputs = layers.Input((self.h.image_size, self.h.image_size, 3))

        # Encoder
        enc_block1 = self.encoder_block(inputs, self.h.n_filters, self.h.batch_norm, dropout_prob=self.h.dropout[0])
        enc_block2 = self.encoder_block(enc_block1[0], self.h.n_filters*2, self.h.batch_norm, dropout_prob=self.h.dropout[1])
        enc_block3 = self.encoder_block(enc_block2[0], self.h.n_filters*4, self.h.batch_norm, dropout_prob=self.h.dropout[2])
        enc_block4 = self.encoder_block(enc_block3[0], self.h.n_filters*8, self.h.batch_norm, dropout_prob=self.h.dropout[3])

        # Bridge
        bridge = self.conv_block(enc_block4[0], self.h.n_filters*16, dropout_prob=self.h.dropout[4])

        # Decoder
        dec_block4 = self.decoder_block(bridge, enc_block4[1], self.h.n_filters*8, self.h.batch_norm, dropout_prob=self.h.dropout[5])
        dec_block3 = self.decoder_block(dec_block4, enc_block3[1], self.h.n_filters*4, self.h.batch_norm, dropout_prob=self.h.dropout[6])
        dec_block2 = self.decoder_block(dec_block3, enc_block2[1], self.h.n_filters*2, self.h.batch_norm, dropout_prob=self.h.dropout[7])
        dec_block1 = self.decoder_block(dec_block2, enc_block1[1], self.h.n_filters, self.h.batch_norm, dropout_prob=self.h.dropout[8])

        conv10 = layers.Conv2D(1, 1, padding='same')(dec_block1)
        output = layers.Activation('sigmoid')(conv10)

        model = tf.keras.Model(inputs=inputs, outputs=output, name='Unet')
        return model

    def read_image(self, image_path):
        # Read image from file in correct format
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) 
        return image
    
    def normalize_input_img(self, image):
        # Normalize input image to range [0, 1]
        image = image.astype('float32') / 255.0
        image = cv.resize(image, (self.h.image_size, self.h.image_size), cv.INTER_NEAREST)
        image = np.reshape(image, (1, self.h.image_size, self.h.image_size, 3))
        return image


    def normalize_output(self, image: np.ndarray):
        # Normalize output image to range [0, 255] as int64
        image = np.rint(image * 255)
        image = image.astype('int64')
        return image

    def forecast(self, input_image):
        # Perform prediction on input image
        input_image = self.normalize_input_img(input_image)

        prediction = self.model.predict(input_image)

        prediction = self.normalize_output(prediction)
        prediction = np.reshape(prediction, (self.h.image_size, self.h.image_size, 1))

        return prediction

if __name__ == '__main__':
    path = 'input/train_v2/'
    # Example usage
    unit = unet_vision()

    images_in_test = os.listdir(path)
    random.shuffle(images_in_test)

    # Get only 10 images to test
    images_in_test = images_in_test[:10]

    for index, image in enumerate(images_in_test):
        print(f'Now image number - {index}')

        image = unit.read_image(path + image)
        result = unit.forecast(image)

        fig, axe = plt.subplots(1, 2)
        
        axe[0].imshow(result, interpolation='nearest')
        axe[1].imshow(image, interpolation='nearest')

        plt.show()

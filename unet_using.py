import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
import random

def conv_block(inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
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

def encoder_block(inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
    # Encoder block with convolutional block and max pooling
    skip_connection = conv_block(inputs, n_filters, batch_norm, dropout_prob)
    next_layer = layers.MaxPooling2D((2, 2))(skip_connection)
    return next_layer, skip_connection

def decoder_block(expansive_input, skip_connection, n_filters, batch_norm=False, dropout_prob=0):
    # Decoder block with transpose convolution and concatenation
    up = layers.Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
    merge = layers.concatenate([up, skip_connection], axis=-1)
    conv = conv_block(merge, n_filters, batch_norm, dropout_prob)
    return conv

def unet_model(input_size=(256, 256, 1), n_filters=64, batch_norm=True, dropouts=np.zeros(9)):
    # U-Net model definition
    inputs = layers.Input(input_size)

    # Encoder
    enc_block1 = encoder_block(inputs, n_filters, batch_norm, dropout_prob=dropouts[0])
    enc_block2 = encoder_block(enc_block1[0], n_filters*2, batch_norm, dropout_prob=dropouts[1])
    enc_block3 = encoder_block(enc_block2[0], n_filters*4, batch_norm, dropout_prob=dropouts[2])
    enc_block4 = encoder_block(enc_block3[0], n_filters*8, batch_norm, dropout_prob=dropouts[3])

    # Bridge
    bridge = conv_block(enc_block4[0], n_filters*16, dropout_prob=dropouts[4])

    # Decoder
    dec_block4 = decoder_block(bridge, enc_block4[1], n_filters*8, batch_norm, dropout_prob=dropouts[5])
    dec_block3 = decoder_block(dec_block4, enc_block3[1], n_filters*4, batch_norm, dropout_prob=dropouts[6])
    dec_block2 = decoder_block(dec_block3, enc_block2[1], n_filters*2, batch_norm, dropout_prob=dropouts[7])
    dec_block1 = decoder_block(dec_block2, enc_block1[1], n_filters, batch_norm, dropout_prob=dropouts[8])

    conv10 = layers.Conv2D(1, 1, padding='same')(dec_block1)
    output = layers.Activation('sigmoid')(conv10)

    model = keras.Model(inputs=inputs, outputs=output, name='Unet')
    return model

def normalize_input_img(image):
    # Normalize input image to range [0, 1]
    if type(image) == np.ndarray:
        return image.astype('float32') / 255.0
    else:
        return tf.cast(image, tf.float32) / 255.0

def normalize_output(image: np.ndarray):
    # Normalize output image to range [0, 255] as int64
    image = np.rint(image * 255)
    image = image.astype('int64')
    return image

class UNet:
    def __init__(self, weight_file_path='training/cp.ckpt', input_size=(256, 256, 1), n_filters=64, batch_norm=True, dropouts=np.zeros(9), grayscale=True):
        self.unet = unet_model(input_size, n_filters, batch_norm, dropouts)
        self.unet.load_weights(weight_file_path).expect_partial()
        self.input_size = input_size
        self.grayscale = grayscale

    def read_image(self, image_path: str):
        # Read and preprocess image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.input_size[:2], method='nearest')

        if self.grayscale:
            image = tf.image.rgb_to_grayscale(image)

        return image

    def forecast(self, input_image):
        # Perform prediction on input image
        input_image = normalize_input_img(input_image)
        input_image = tf.expand_dims(input_image, axis=0)
        prediction = self.unet.predict(input_image)
        prediction = normalize_output(prediction)
        return prediction

if __name__ == '__main__':
    path = 'airbus-ship-detection/test/'
    # Example usage
    unit = UNet(input_size=(128, 128, 3), weight_file_path='end_training_weights_dice_loss/end_weight', n_filters=32, grayscale=False)

    images_in_test = os.listdir(path)
    random.shuffle(images_in_test)

    # Get only 100 images to test
    images_in_test = images_in_test[:100]

    for index, image in enumerate(images_in_test):
        print(f'Now image number - {index}')
        image = unit.read_image(path + image)
        result = unit.forecast(image)

        result = np.reshape(result, (128, 128, 1))
        image = np.reshape(image, (128, 128, 3))

        fig, axe = plt.subplots(1, 2)
        axe[0].imshow(result, interpolation='nearest')
        axe[1].imshow(image, interpolation='nearest')
        plt.show()

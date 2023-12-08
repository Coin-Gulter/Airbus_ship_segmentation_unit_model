import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import parameters
import preprocesing

class unet_train_model():
    def __init__(self):
        self.h = parameters.get_config()

        self.preprocess = preprocesing.preprocesing_ds(self.h)

        self.model = self.create_unet_model()
        
        if self.h.pretrained:
            self.model.load_weights(os.path.join(self.h.save_model_folder, self.h.save_model_name))
            

    # Function to define a convolutional block
    def conv_block(self, inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
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

    # Function to define an encoder block
    def encoder_block(self, inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
        skip_connection = self.conv_block(inputs, n_filters, batch_norm, dropout_prob)
        next_layer = layers.MaxPooling2D((2,2))(skip_connection)
        return next_layer, skip_connection

    # Function to define a decoder block
    def decoder_block(self, expansive_input, skip_connection, n_filters, batch_norm=False, dropout_prob=0):
        up = layers.Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
        
        merge = layers.concatenate([up, skip_connection], axis=-1)

        conv = self.conv_block(merge, n_filters, batch_norm, dropout_prob)
        return conv

    # Function to define the U-Net model
    def create_unet_model(self):
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

        conv10 = layers.Conv2D(self.h.num_classes, 1, padding='same')(dec_block1)
        output = layers.Activation(self.h.act_type)(conv10)

        model = keras.Model(inputs=inputs, outputs=output, name='Unet')
        return model


    def dice_score(self, y_true, y_pred):
        smooth = 1e-6  # Smoothing factor to avoid division by zero

        # Flatten the true and predicted labels
        y_true_flat = tf.keras.backend.flatten(y_true)
        y_pred_flat = tf.keras.backend.flatten(y_pred)
        y_true_flat = tf.cast(y_true_flat, dtype=tf.float32)

        # Calculate the intersection and union
        intersection = tf.keras.backend.sum(y_true_flat * y_pred_flat)
        union = tf.keras.backend.sum(y_true_flat) + tf.keras.backend.sum(y_pred_flat)

        # Calculate the Dice score
        dice = (2 * intersection + smooth) / (union + smooth)

        return dice

    def dice_loss(self, y_true, y_pred):
        return 1 - self.dice_score(y_true, y_pred)

    def true_positive_rate(self, y_true, y_pred):
        return tf.keras.backend.sum(tf.keras.backend.flatten(y_true) * tf.keras.backend.flatten(tf.keras.backend.round(y_pred))) / tf.keras.backend.sum(y_true)


    def train(self):
        ds_train, ds_valid = self.preprocess.create_ds()

        # Define a callback to save the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.h.checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.h.learning_rate), loss=self.dice_loss, 
                           metrics=[self.dice_score, 'accuracy', self.true_positive_rate])

        seqModel = self.model.fit(ds_train, epochs=self.h.num_epochs, validation_data=ds_valid, callbacks=[cp_callback])

        self.model.save_weights(self.h.save_model_folder + self.h.save_model_name)

        if self.h.show_training_plot:
            # Retrieve the loss and metrics history
            train_loss = seqModel.history['loss']
            val_loss = seqModel.history['val_loss']
            train_dice = seqModel.history['dice_score']
            val_acc = seqModel.history['val_dice_score']
            xc = range(self.h.num_epochs)

            # Plot the loss history
            plt.figure()
            plt.plot(xc, train_loss, 'b-', label='train_loss')
            plt.plot(xc, train_dice, 'r-', label='train_dice')
            plt.plot(xc, val_loss, 'c-', label='val_loss')
            plt.plot(xc, val_acc, 'm-', label='val_acc')
            plt.legend()
            plt.show()


if __name__ == "__main__":
    unet_train = unet_train_model()

    unet_train.train()

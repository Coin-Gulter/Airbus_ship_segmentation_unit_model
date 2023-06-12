import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import numpy as np

# Function to read and decode image
def read_image(image_path, label, path_folder, grayscale=True):
    image = tf.io.read_file(path_folder + image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    if grayscale:
        image = tf.image.rgb_to_grayscale(image)
    return image, label

# Function to normalize image
def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label

# Function to rescale image and label
def rescale_data(image, label, new_size=(256,256)):
    image = tf.image.resize(image, new_size, method='nearest')
    label = tf.image.resize(label, new_size, method='nearest')
    return image, label

# Function to decode the Run-Length Encoding (RLE) label
def rle_decode_tf(image_path, label, shape):
    shape = tf.convert_to_tensor(shape, tf.int64)
    size = tf.math.reduce_prod(shape,)
    # Split string
    s = tf.strings.split(label)
    s = tf.strings.to_number(s, tf.int64)
    # Get starts and lengths
    starts = s[::2] - 1
    lens = s[1::2]
    # Make ones to be scattered
    total_ones = tf.reduce_sum(lens)
    ones = tf.ones([total_ones], tf.uint8)
    # Make scattering indices
    r = tf.range(total_ones)
    lens_cum = tf.math.cumsum(lens)
    s = tf.searchsorted(lens_cum, r, 'right')
    idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)
    # Scatter ones into flattened mask
    label = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
    # Reshape into mask
    label = tf.reshape(label, shape)
    # Rotate mask to 270 degree counter clockwise 
    label = tf.image.rot90(label, k=3)
    label = tf.image.flip_left_right(label)

    return image_path, label

# Function to create the dataset
def create_ds(path_folder='airbus-ship-detection/train/', csv_file='train_ship_segmentations.csv', elements_number=0, valid_persentage=0.1, label_shape=(768,768,1), label_reshape=(128,128), grayscale=True, cache=True, shuffle=1000, batch_size=64, prefetch=1):
    print('Creating dataset .....')
    full_df = pd.read_csv(path_folder+csv_file)

    full_df = full_df.dropna()
    full_df = full_df.groupby('ImageId')['EncodedPixels'].agg(' '.join).reset_index(name='EncodedPixels')

    if elements_number != 0:
        train_df = full_df.head(elements_number)
        valid_df = full_df.tail(int(elements_number*valid_persentage))
    else:
        length = full_df.shape[0]
        train_df = full_df.head(int(length*(1-valid_persentage)))
        valid_df = full_df.tail(int(length*valid_persentage))

    labels = train_df['EncodedPixels'].values
    train_files = train_df['ImageId'].values
    val_labels = valid_df['EncodedPixels'].values
    val_files = valid_df['ImageId'].values

    ds_train = tf.data.Dataset.from_tensor_slices((train_files, labels))
    ds_valid = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    # Apply preprocessing functions to the dataset
    ds_train = ds_train.map(lambda x, y: rle_decode_tf(x, y, label_shape)).map(lambda v, b: read_image(v, b, path_folder, grayscale=grayscale)).map(normalize_img).map(lambda i, j: rescale_data(i, j, label_reshape))
    ds_valid = ds_valid.map(lambda x, y: rle_decode_tf(x, y, label_shape)).map(lambda v, b: read_image(v, b, path_folder, grayscale=grayscale)).map(normalize_img).map(lambda i, j: rescale_data(i, j, label_reshape))

    if cache:
        ds_train = ds_train.cache()
        ds_valid = ds_valid.cache()
    if shuffle:
        ds_train = ds_train.shuffle(shuffle)
    if batch_size:
        ds_train = ds_train.batch(batch_size)
        ds_valid = ds_valid.batch(batch_size)
    if prefetch:
        ds_train = ds_train.prefetch(prefetch)

    print('Dataset created .....')
    return ds_train, ds_valid

# Function to define a convolutional block
def conv_block(inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
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
def encoder_block(inputs=None, n_filters=64, batch_norm=False, dropout_prob=0):
    skip_connection = conv_block(inputs, n_filters, batch_norm, dropout_prob)
    next_layer = layers.MaxPooling2D((2,2))(skip_connection)
    return next_layer, skip_connection

# Function to define a decoder block
def decoder_block(expansive_input, skip_connection, n_filters, batch_norm=False, dropout_prob=0):
    up = layers.Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
    merge = layers.concatenate([up, skip_connection], axis=-1)
    conv = conv_block(merge, n_filters, batch_norm, dropout_prob)
    return conv

# Function to define the U-Net model
def unet_model(input_size=(768, 768, 1), n_filters=64, batch_norm=True, dropouts=np.zeros(9)):
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


def dice_score(y_true, y_pred):
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

def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)

def true_positive_rate(y_true, y_pred):
    return tf.keras.backend.sum(tf.keras.backend.flatten(y_true) * tf.keras.backend.flatten(tf.keras.backend.round(y_pred))) / tf.keras.backend.sum(y_true)

TRAIN_DIR = 'airbus-ship-detection/train/'
CHECKPOINT_PATH = "checkpoint_dice_loss/cp.ckpt"
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 10

if __name__ == "__main__":
    # Create the training and validation datasets
    ds_train, ds_valid = create_ds(path_folder=TRAIN_DIR, csv_file='train_ship_segmentations.csv', elements_number=0, label_reshape=(128, 128),
                                   grayscale=False, cache=True, shuffle=10000, batch_size=24, prefetch=False)

    # Create the U-Net model
    unet = unet_model(input_size=(128, 128, 3), n_filters=32, dropouts=[0.3]*9)
    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=dice_loss, metrics=[dice_score, 'accuracy', true_positive_rate])

    # Define a callback to save the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train the model
    seqModel = unet.fit(ds_train, epochs=EPOCHS, validation_data=ds_valid, callbacks=[cp_callback])
    
    # Save the final weights
    unet.save_weights('end_training_weights_dice_loss/end_weight')

    # Retrieve the loss and metrics history
    train_loss = seqModel.history['loss']
    val_loss = seqModel.history['val_loss']
    train_dice = seqModel.history['dice_score']
    val_acc = seqModel.history['val_dice_score']
    xc = range(EPOCHS)

    # Plot the loss history
    plt.figure()
    plt.plot(xc, train_loss, 'b-', label='train_loss')
    plt.plot(xc, train_dice, 'r-', label='train_dice')
    plt.plot(xc, val_loss, 'c-', label='val_loss')
    plt.plot(xc, val_acc, 'm-', label='val_acc')
    plt.legend()
    plt.show()

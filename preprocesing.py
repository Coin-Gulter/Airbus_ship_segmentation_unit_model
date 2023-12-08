import pandas as pd
import numpy as np
import imutils
import re
import cv2 as cv
import tensorflow as tf
import parameters


class preprocesing_ds():

    def __init__(self, h_parameters):
        # Constructor for the preprocessing class, responsible for data preparation.
        self.h = h_parameters

    def rle_decode(self, label):
        # Function to decode the Run-Length Encoding (RLE) label
        size = self.h.input_image_size*self.h.input_image_size
        mask = np.zeros(size)

        label = re.sub(r'[^0-9\s]', '', label)
        label = label.split()

        for index in range(0, len(label), 2):
            start_pixel = int(label[index])
            end_pixel = int(label[index]) + int(label[index+1])
            mask[start_pixel:end_pixel] = 1

        mask = np.reshape(mask, (self.h.input_image_size, self.h.input_image_size))

        mask = cv.resize(mask, (self.h.image_size, self.h.image_size), interpolation=cv.INTER_NEAREST)
        mask = cv.flip(mask, 1)
        mask = imutils.rotate(mask, 90)

        mask = np.reshape(mask, (self.h.image_size, self.h.image_size, 1))

        return mask

    def read_image_tf(self, image_path, label, path_folder):
        # Function to read and decode image
        image = tf.io.read_file(path_folder + image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image, label

    def normalize_img(self, image, label):
        # Function to normalize image
        return tf.cast(image, tf.float32)/255.0, label

    def rescale_image(self, image, label, image_size):
        # Function to rescale image and label
        image = tf.image.resize(image, image_size, method='nearest')
        return image, label


    def create_ds(self):
        # Function to create the dataset

        print('Creating dataset .....')

        full_df = pd.read_csv(self.h.train_path+self.h.train_file)

        full_df = full_df.dropna()
        full_df = full_df.groupby('ImageId')['EncodedPixels'].agg(' '.join).reset_index(name='EncodedPixels')

        if self.h.number_train_images != 0:
            train_df = full_df.head(self.h.number_train_images)
            valid_df = full_df.tail(int(self.h.number_train_images*self.h.validation_split))
        else:
            length = full_df.shape[0]
            train_df = full_df.head(int(length*(1-self.h.validation_split)))
            valid_df = full_df.tail(int(length*self.h.validation_split))

        labels = train_df['EncodedPixels'].values
        train_files = train_df['ImageId'].values
        val_labels = valid_df['EncodedPixels'].values
        val_files = valid_df['ImageId'].values

        labels = np.array(list(map(self.rle_decode, labels)))
        val_labels = np.array(list(map(self.rle_decode, val_labels)))

        ds_train = tf.data.Dataset.from_tensor_slices((train_files, labels))
        ds_valid = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

        # Apply preprocessing functions to the dataset
        ds_train = ds_train.map(lambda v, b: self.read_image_tf(v, b, self.h.train_path))
        ds_train = ds_train.map(self.normalize_img)
        ds_train = ds_train.map(lambda i, j: self.rescale_image(i, j, (self.h.image_size, self.h.image_size)))
        
        ds_valid = ds_valid.map(lambda v, b: self.read_image_tf(v, b, self.h.train_path))
        ds_valid = ds_valid.map(self.normalize_img)
        ds_valid = ds_valid.map(lambda i, j: self.rescale_image(i, j, (self.h.image_size, self.h.image_size)))

        if self.h.cache:
            ds_train = ds_train.cache()
            ds_valid = ds_valid.cache()
        if self.h.shuffle:
            ds_train = ds_train.shuffle(self.h.shuffle)
        if self.h.batch_size:
            ds_train = ds_train.batch(self.h.batch_size)
            ds_valid = ds_valid.batch(self.h.batch_size)
        if self.h.prefetch:
            ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        print('Dataset created .....')

        return ds_train, ds_valid



if __name__ == '__main__':
    # Preprocessing test
    h_parameters = parameters.get_config()

    preproc = preprocesing_ds(h_parameters)

    print(preproc.create_ds())

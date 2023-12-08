# U-Net Training and Usage

This solution provides code for training and using a U-Net model for image segmentation tasks. The U-Net architecture is a popular choice for tasks such as medical image segmentation and semantic segmentation. 
In this solution image segmentation using to identified ship in image from satelites. Repository has two file to train ("unet_train.py") and using ("unet_vision.py) U-Net model, 
()"exploratory_data.ipynb") with airbus-ship-detection dataset analysis and two folder `weights` and `checkpoint_dice_loss` with trained weigth and checkpoint weights for parameters input_size=(128,128,3) and n_filters=32. 

Model with pretrained weight has dice score precision up to 75%. 

To test or train you can use dataset from popular segmentation problem `Airbus Ship Detection Challenge` from kaggel using link `https://www.kaggle.com/c/airbus-ship-detection/data`. 
To train using `Airbus Ship Detection Challenge`, put dataset folder in project folder and rename as "input". After move `airbus-ship-detection.csv` training file from upper folder to "train_v2" folder.

## Training the U-Net Model

The "unet_train.py" script provides functions and code for training the U-Net model. 
It using "parameters.py" to get parameters of model and tranining and "preprocessing.py" to prepare dataset to training. Here are the main components of the script:

### Dataset Preparation "preprocessing.py"

- `read_image`: Method to read and decode image files.
- `normalize_img`: Method to normalize image pixel values.
- `rescale_data`: Method to rescale image and label to a desired size.
- `rle_decode_tf`: Method to decode Run-Length Encoding (RLE) labels.
- `create_ds` : Method to create ended training dataset. It returns two values (ds_train, ds_valid) dataset for training and validation.

### U-Net Model Definition ("unet_train.py", "unet_vision.py")

- `conv_block`: Function to define a convolutional block.
- `encoder_block`: Function to define an encoder block.
- `decoder_block`: Function to define a decoder block.
- `unet_model`: Function to define the U-Net model.

### Loss and Metrics

- `dice_score`: Function to calculate the Dice coefficient for model evaluation.
- `dice_loss`: Function to calculate the Dice loss for model training.
- `true_positive_rate`: Function to calculate the true positive rate.

### Training

- `TRAIN_DIR`: Training data directory.
- `AUTOTUNE`: Constant autotuning using for prefetch data.
- `EPOCHS`: Number of training epochs.

To train the U-Net model, run the `unet_train.py` script.

Here some example how simple it is:

    if __name__ == "__main__":
        unet_train = unet_train_model()

        unet_train.train()

To get better training result we can make image size for training bigger than 128 in "image_size = 128", increase number of train in "number_train_images = 2000", change number of epochs "num_epochs = 12" and so on in "parameters.py".

Here example how it looks:

    page_model_param_dict = {
        'unet_segmentation':
            dict(
                number_train_images = 2000,
                image_size = 128,
                num_epochs = 12,
                pretrained = False,
                learning_rate = 0.01,
                validation_split = 0.1,
                batch_size = 16,
            )
    }

## Using the Trained U-Net Model

The `unet_vision.py` script provides code for using the trained U-Net model. Here are the main components of the script:

- `conv_block`: Function to define a convolutional block.
- `encoder_block`: Function to define an encoder block.
- `decoder_block`: Function to define a decoder block.
- `unet_model`: Function to define the U-Net model.
- `normalize_input_img`: Function to normalize input image pixel values.
- `normalize_output`: Function to normalize output image pixel values.
- class `UNet`: Class to using U-Net model.
- class `UNet`-`read_image`: function to read image to predict result.
- class `UNet`-`forecast`: funtion to predict U-Net result of image.

To test the U-Net model, run the `unet_vision.py` script.

example how you could use code shown below:

    import random
    import os
    from matplotlib import pyplot as plt
    import unet_vision


    path = 'input/train_v2/'

    images_in_test = os.listdir(path)
    random.shuffle(images_in_test)

    # Get only 10 images to test
    images_in_test = images_in_test[:10]


    # Example usage
    unit = unet_vision.unet_vision()

    for index, image in enumerate(images_in_test):
        print(f'Now image number - {index}')

        #usage of unit class
        image = unit.read_image(path + image)
        result = unit.forecast(image)

        #Show result
        fig, axe = plt.subplots(1, 2)
        axe[0].imshow(result, interpolation='nearest')
        axe[1].imshow(image, interpolation='nearest')
        plt.show()

To predict using pretrained model use method 'forecast()' from class 'unet_vision'. As argumnets it takes 'image' in numpy format of shape (128, 128, 3) better in RGB sequence or if you have image as .jpg .png file you can use method 'read_image()' of class 'unet_vision' and get you image in corect format.

# Thank you for attention (^_^)

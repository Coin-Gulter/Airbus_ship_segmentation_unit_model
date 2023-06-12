# U-Net Training and Usage

This solution provides code for training and using a U-Net model for image segmentation tasks. The U-Net architecture is a popular choice for tasks such as medical image segmentation and semantic segmentation. In this solution image segmentation using to 
identified ship in image from satelites. Repository has two file to train and using U-Net model, exploratory_data.ipynb with airbus-ship-detection dataset analysis and two folder end_training_weight_dice_loss checkpoint_dice_loss with trained weigth for input_size=(128,128,3) and n_filters=32. Model with pretrained weight has precision up to 75%. To test or train you can use airbus-ship-detection dataset from kaggel using link `https://www.kaggle.com/c/airbus-ship-detection/data`.

## Requirements

The solution requires the following dependencies:

- Python 3.7 or higher
- TensorFlow 2.0 or higher
- NumPy
- pandas
- matplotlib
- random
- os

## Training the U-Net Model

The `unet_train.py` script provides functions and code for training the U-Net model. Here are the main components of the script:

### Dataset Preparation

- `read_image`: Function to read and decode image files.
- `normalize_img`: Function to normalize image pixel values.
- `rescale_data`: Function to rescale image and label to a desired size.
- `rle_decode_tf`: Function to decode Run-Length Encoding (RLE) labels.

### Dataset Creation

- `create_ds`: Function to create the training and validation datasets.

### U-Net Model Definition

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

## Using the Trained U-Net Model

The `unet_usage.py` script provides code for using the trained U-Net model. Here are the main components of the script:

- `conv_block`: Function to define a convolutional block.
- `encoder_block`: Function to define an encoder block.
- `decoder_block`: Function to define a decoder block.
- `unet_model`: Function to define the U-Net model.
- `normalize_input_img`: Function to normalize input image pixel values.
- `normalize_output`: Function to normalize output image pixel values.
- class `UNet`: Class to using U-Net model.
- class `UNet`-`read_image`: function to read image to predict result.
- class `UNet`-`forecast`: funtion to predict U-Net result of image.

To test the U-Net model, run the `unet_usage.py` script.


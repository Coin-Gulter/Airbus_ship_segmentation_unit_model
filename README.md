# U-Net Training and Usage

This solution provides code for training and using a U-Net model for image segmentation tasks. The U-Net architecture is a popular choice for tasks such as medical image segmentation and semantic segmentation. In this solution image segmentation using to 
identified ship in image from satelites. 

## Requirements

The solution requires the following dependencies:

- Python 3.7 or higher
- TensorFlow 2.0 or higher
- NumPy
- pandas
- matplotlib

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

To use the trained U-Net model, run the `unet_usage.py` script.


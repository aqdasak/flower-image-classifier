# Flower Image Classifier CLI

This project consists of a pair of Python scripts that run from the command line to train a neural network on a dataset of flower images and use it to predict the class of new images.

## Files Included

- `train.py`: trains a new network on a given dataset and saves the trained model as a checkpoint
- `predict.py`: uses a trained network to predict the class for a given input image
- `cat_to_name.json`: maps category labels to flower names for use in prediction output

## Requirements

To run this project, you will need:

- Python 3.x
- PyTorch
- NumPy
- PIL
- argparse

## Usage

Create a new conda environment with required dependencies
```
conda create --name <env> --file requirements.txt
```

### Training

To train a new network on a dataset, run `train.py` with the following command:
```
python train.py data_directory
```
The `data_directory` argument should be the path to the directory containing the image data, which should be organized into subdirectories according to category.

The script will print out training loss, validation loss, and validation accuracy as the network trains. Optional arguments can be used to customize the training process, including:

- `--save_dir`: set directory to save checkpoints
- `--arch`: choose architecture [vgg11(default) or vgg13]
- `--learning_rate`: set learning rate
- `--hidden_units`: set number of hidden units in the network
- `--epochs`: set number of training epochs
- `--gpu`: use GPU for training

### Prediction

To use a trained network to predict the class for a new image, run `predict.py` with the following command:
```
python predict.py /path/to/image checkpoint
```
The `image` argument should be the path to the image file, and `checkpoint` should be the path to the saved model checkpoint.

The script will print out the predicted flower name and class probability. Optional arguments can be used to customize the prediction output, including:

- `--top_k`: return top k most likely classes
- `--category_names`: use a mapping of categories to real names (e.g. cat_to_name.json)
- `--gpu`: use GPU for inference

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgement

This project was completed as part of the Udacity 'AI Programming with Python' Nanodegree program.
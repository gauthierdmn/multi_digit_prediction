# Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks

The goal of this project was to create end-to-end Deep Convolutional Neural Networks applied on the Street View House Numbers dataset to classify multi-digit numbers.

All you need to run the code is Python 3.x and Tensorflow.

Note that I trained the models on GPUs to make the training faster so running in local may take some time when training the model.

## File Description

* `DigitSequence`: a folder with Python files to download the SVHN dataset and train the final model. Just python type `python train.py` in this repository using your terminal to train the model.
* `MNIST_multi_digit_classification.ipynb`: a notebook where a Deep CNN is trained on multi-digit numbers from MNIST dataset (toy exemple)
* `SVHN_multi_digit_classification.ipynb`: a notebook where a Deep CNN is trained on multi-digit numbers from SVHN dataset
* `DigitStructFile.py`: a Python file containing helper functions to deal with .mat files

## Resources

* [Street View House Number (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)
* [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf)


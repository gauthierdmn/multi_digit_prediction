from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import os
from six.moves.urllib.request import urlretrieve
import sys
import tarfile
import PIL.Image as Image
from DigitStructFile import *


FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_integer('IM_SIZE', 32,
 #                           """Size of cropped SVHN images.""")
tf.app.flags.DEFINE_integer('num_channels', 1,
                            """Number of channels: 1 if B&W else 3.""")

def maybe_download(url, filename, dirc=""):
    if not os.path.exists(dirc + filename):
        print ('Attempting to download: ' + filename)
        filename, _ = urlretrieve(url + filename, dirc + filename)
        print ('\nDownload Complete!')
    else:
        filename = dirc + filename
        print ('Found and verified', filename)
    return filename

def create_svhn(data, folder):
    dataset = np.ndarray([int(len(data) * 0.8), FLAGS.IM_SIZE, FLAGS.IM_SIZE], dtype='float32')
    labels = np.ndarray([int(len(data) * 0.8), 5], dtype='int32')
    length = np.ndarray([int(len(data) * 0.8)], dtype='int32')

    for i in np.arange(int(len(data) * 0.8)):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        im = Image.open(fullname)
        num_digit = len(data[i]['boxes'])

        if num_digit > 5:
            continue

        # Get the expanded bounding box by 30%
        im_top, im_left, im_height, im_width = increase_bounding_box(data[i], im.size, 0.3)

        # Cropping the expanded bounding box
        im = im.crop((int(im_left), int(im_top), int(im_left + im_width), int(im_top + im_height))).resize(
            [FLAGS.IM_SIZE, FLAGS.IM_SIZE], Image.ANTIALIAS)
        # Transform into grayscale
        im = (np.array(im)[:, :, 0] * 0.299 + np.array(im)[:, :, 1] * 0.587 + np.array(im)[:, :, 2] * 0.114) / 3
        dataset[i, :, :] = im

        for digit in range(num_digit):
            temp_label = data[i]['boxes'][digit]['label']
            labels[i, digit] = temp_label
        for digit in range(5 - num_digit):
            labels[i, 4 - digit] = 0
        length[i] = int(num_digit - 1)

    dataset = dataset.reshape((-1, FLAGS.IM_SIZE, FLAGS.IM_SIZE, 1)).astype(np.float32)

    return dataset, labels, length

def load_svhn():
    train_folders = 'train'
    test_folders = 'test'

    if not os.path.exists('temp_train.pickle') and not os.path.exists('temp_test.pickle'):
        train_filename = maybe_download('http://ufldl.stanford.edu/housenumbers/', 'train.tar.gz')
        test_filename = maybe_download('http://ufldl.stanford.edu/housenumbers/', 'test.tar.gz')

        extract_afile(train_filename)
        extract_afile(test_filename)

        fin = os.path.join(train_folders, 'digitStruct.mat')
        dsf = DigitStructFile(fin)
        train_data = dsf.getAllDigitStructure_ByDigit()

        print('Train ready')

        fin = os.path.join(test_folders, 'digitStruct.mat')
        dsf = DigitStructFile(fin)
        test_data = dsf.getAllDigitStructure_ByDigit()

        print('Test ready')


        pickle_file = 'temp_train.pickle'

        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

        pickle_file = 'temp_test.pickle'

        try:
            f = open(pickle_file, 'wb')
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    pickle_name = 'temp_train.pickle'

    with open(pickle_name, 'rb') as f:
        train_data = pickle.load(f)

    pickle_name = 'temp_test.pickle'

    with open(pickle_name, 'rb') as f:
        test_data = pickle.load(f)

    train_dataset, train_labels, train_length = create_svhn(train_data, train_folders)
    print(train_dataset.shape, train_labels.shape, train_length.shape)
    del train_data

    test_dataset, test_labels, test_length = create_svhn(test_data, test_folders)
    print(test_dataset.shape, test_labels.shape, train_length.shape)
    del test_data

    train_dataset = subtract_mean(train_dataset)
    test_dataset = subtract_mean(test_dataset)

    return [train_dataset, test_dataset, train_labels, test_labels, train_length, test_length]

def reformat_dataset(dataset):
    dataset = dataset.reshape(
    (-1, FLAGS.image_h, FLAGS.image_w, FLAGS.num_channels)).astype(np.float32)
    return dataset

def maybe_pickle(set_filename, data):
    try:
        with open(set_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

def extract_afile(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root):
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = root
    print(data_folders)


def get_bounding_box(meta_dict):
    num_digit = len(meta_dict['boxes'])
    boxes = meta_dict['boxes']

    # Get the individual bounding boxes
    top = np.ndarray([num_digit], dtype='float32')
    left = np.ndarray([num_digit], dtype='float32')
    height = np.ndarray([num_digit], dtype='float32')
    width = np.ndarray([num_digit], dtype='float32')

    for j in np.arange(num_digit):
        top[j] = boxes[j]['top']
        left[j] = boxes[j]['left']
        height[j] = boxes[j]['height']
        width[j] = boxes[j]['width']

    # Get the bounding box surrounding all digits
    im_top = np.amin(top)
    im_left = np.amin(left)
    im_height = np.amax(top) + height[np.argmax(top)] - im_top
    im_width = np.amax(left) + width[np.argmax(left)] - im_left

    return im_top, im_left, im_height, im_width


def increase_bounding_box(meta_dict, im_size, increase_factor):
    im_top, im_left, im_height, im_width = get_bounding_box(meta_dict)
    new_height = np.amin([np.ceil((1 + increase_factor) * im_height), im_size[1]])
    new_width = np.amin([np.ceil((1 + increase_factor) * im_width), im_size[0]])
    new_left = np.amax([np.floor(im_left - increase_factor * im_width / 2), 0])
    new_top = np.amax([np.floor(im_top - increase_factor * im_height / 2), 0])
    return new_top, new_left, new_height, new_width

def subtract_mean(a):
    a = (a - np.mean(a, axis=0)) / np.std(a, axis=0)
    return a
"""
Functions for loading in and working with our data
and doing initial visualization
"""
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2

import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import image_dataset_from_directory

def load_images_dataset(image_dir,
                        target_size=(240,320),
                        batch_size=32,
                        validation_split=None):
    """
    Creates a dataset to load in our images into our models

    Parameters
    ----------
    dir: path to images
    target_size: size that images will be resized to
    batch_size: batch size
    validation_split: percentage of data to be split for validation

    Returns
    -------
    (train_data, val_data): tuple of training dataset, validation dataset if validation_split
    dataset: other wise
    """
    if validation_split is not None:
        train_data, val_data = image_dataset_from_directory(image_dir,
                                                           batch_size=batch_size,
                                                           image_size=target_size,
                                                           seed=123,
                                                           validation_split=validation_split,
                                                           subset='both')
        return train_data, val_data
    else:
        return image_dataset_from_directory(image_dir,
                                            batch_size=batch_size,
                                            image_size=target_size)

#############################
#No longer in use:
#############################

def load_images_resize(image_dir, target_size, labels_to_rep):
    """
    Loads the images at the directory path dir into an ndarray, then returns it
    with a corresponding list of training labels (from the directory name) and
    the names of the image files.

    Parameters
    ----------
    image_dir: path to image directory
    target_size: size images should be resized to
    labels_to_rep: dict with keys: true labels from directory,
                             values: corresponding integer representation

    Returns
    -------
    images: ndarray of size (num images, 240, 320, 3)
    images_names: list of len (num images)
    labels: list of len (num images)
    """
    # create empty lists to store image data
    images_list = [] # ndarrays for the training images
    images_names = [] # the filename of the image
    labels = [] # the true label for the image

    # loop through directory:
    for dirpath, dirnames, filenames in os.walk(image_dir):
        for file in filenames:
            # append resized image
            images_list.append(cv2.resize(src=iio.imread(uri = os.path.join(dirpath, file)),
                                          dsize=target_size,
                                          interpolation=cv2.INTER_AREA))
            images_names.append(file)
            # assign the image a rep based on the directory it lies in
            labels.append(labels_to_rep[Path(dirpath).stem])

    # stack all images into one array and normalize values to [0,1]
    images = np.stack(images_list, axis=0)/255

    return images, images_names, labels

def batch_resize(images, new_shape):
    """
    Resizes all images into desired shape (by batch).

    Parameters
    ----------
    images: 4D ndarray of images of shape (number images, height, width, channels)
    new_shape: tuple of (new_height, new_width)

    Returns
    -------
    out_images: 4D ndarray of images of shape (number images, new_height, new_width, channels)
    """

    num_images, old_height, old_width, channels = images.shape

    out_images_list = [] # empty array to store each batch

    # resize each image
    for i in range(0, num_images):
        resized_image = cv2.resize(src=images[i, :, :, :].copy(),
                                   dsize=new_shape,
                                   interpolation=cv2.INTER_AREA)

        out_images_list.append(resized_image)

    out_images = np.stack(out_images_list, axis=0)
    return out_images

"""
Functions to visualize our data/ model performance
"""
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

def get_n_images(image_dataset, n_images):
    """
    Retrieve n_images many images and their labels from images_dataset

    Parameters
    ----------
    image_dataset: tf.data.Dataset object with images/labels
    n_images: integer number of images to return

    Returns
    -------
    image_subset: n_images many images, ndarray of shape
                  (n_images, image_height, image_width, channels)
    image_labels: labels corresponding to the n_images many images
                  ndarray of shape (n_images,)

    """

    # convert dataset to list of batch_size many images and their labels
    data_list = list(image_dataset.as_numpy_iterator())

    # extract batchsize from dataset list
    batch_size = data_list[0][0].shape[0]

    # take n images from the dataset
    # depending on the number of images/labels in a batch,
    # read in more batches if necessary
    if batch_size >= n_images:
        image_subset = data_list[0][0][:n_images,:,:,:]
        image_labels = data_list[0][1][:n_images]
    else:
        image_subset_i = []
        image_labels_i = []
        # store more batches until we have enough images
        for i in range(int(np.ceil(n_images/batch_size))):
            image_subset_i.append(data_list[i][0])
            image_labels_i.append(data_list[i][1])
        # create required subset of images/labels
        image_subset = np.concatenate(image_subset_i, axis=0)[:n_images,:,:,:]
        image_labels = np.vstack(image_labels_i)[:n_images, :].reshape((n_images,))

    return image_subset, image_labels

def visualize_dataset(image_dataset, rep_to_labels):
    """
    Plots 10 images from the image_dataset

    Parameters
    ----------
    image_dataset: tf.data.Dataset object with images and image labels
    rep_to_labels: dict with keys: integer representation of true labels
                             values: corresponding labels

    Returns
    -------
    None
    """
    # get images and their corresponding labels
    image_subset, image_labels = get_n_images(image_dataset, 10)

    # plot
    fig, axs = plt.subplots(2, 5, figsize=(20,8))
    count = 0
    for i in range(2):
        for j in range(5):
            axs[i,j].imshow(image_subset[count,:,:,:]/255)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            # axs[i,j].axis("off")
            axs[i,j].set_xlabel(rep_to_labels[image_labels[count]])
            count+=1
    plt.show()

def visualize_history(history):
    """
    Visualizes the training/validation accuracy and loss during training

    Parameters
    ----------
    history: tf History object (e.g. that returned by a keras.fit call)

    Returns
    -------
    None
    """
    # plot training and validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_predictions(model, test_dataset, rep_to_labels):
    """
    Picks 10 test images from test_dataset, and displays them
    along with the model's prediction and the prediction probability

    Parameters
    ----------
    model: model to be used for prediction
    test_dataset: tf.data.Dataset object with test images/labels
    rep_to_labels: dict with keys: integer representation of true labels
                             values: corresponding labels

    Returns
    -------
    None
    """
    # create model that yields prediction probabilites based on model_enet_tl
    model_prob = keras.models.Sequential([model, keras.layers.Softmax()])

    # get 10 images from test_dataset and their true labels
    image_subset, image_true_labels = get_n_images(test_dataset, 10)

    # get prob of predicted labels
    pred_labels_prob = model_prob.predict(image_subset)

    # get predicted labels by choosing the one with highest probability
    pred_labels_index = np.argmax(pred_labels_prob, axis=1)

    image_true_labels = [rep_to_labels[label] for label in image_true_labels]
    pred_labels = [rep_to_labels[label] for label in pred_labels_index]

    # plot the sample images along with their true and predicted labels
    plt.figure(figsize=(20, 8))
    plt.suptitle('Model Predictions on Sample Test Data')
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(image_subset[i]/255)
        plt.xlabel(f"True label: {image_true_labels[i]}\n"
                    f"Predicted label: {pred_labels[i]}\n"
                   f"Probability: {pred_labels_prob[i,pred_labels_index[i]]:.4f}")
        plt.xticks([])
        plt.yticks([])

    # Adjust plots for nicer visualization and display the result
    plt.tight_layout()
    plt.show()

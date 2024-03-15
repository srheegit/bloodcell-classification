"""
Functions to visualize our data/ model performance
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix, classification_report

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

def visualize_history(history, model_name=None):
    """
    Visualizes the training/validation accuracy and loss during training

    Parameters
    ----------
    history: tf History object (e.g. that returned by a keras.fit call)
    model_name: name of model to be displayed as title

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

    if model_name is not None:
        plt.suptitle(f"Model {model_name}")

    plt.tight_layout()
    plt.show()

def visualize_predictions(model,
                          rep_to_labels,
                          test_dataset=None,
                          test_images=None,
                          test_labels=None,
                          model_name=None):
    """
    Picks 10 test images from test_dataset, and displays them
    along with the model's prediction and the prediction probability

    Parameters
    ----------
    model: model to be used for prediction
    rep_to_labels: dict with keys: integer representation of true labels
                             values: corresponding labels
    test_dataset: tf.data.Dataset object with test images/labels
    model_name: name of model to be displayed in title
    test_images: ndarray of 10 images to be displayed
    test_labels: ndarray of 10 labels to be displayed

    Returns
    -------
    None
    """

    # create model that yields prediction probabilites based on model_enet_tl
    model_prob = keras.models.Sequential([model, keras.layers.Softmax()])

    if test_images is not None and test_labels is not None:
        image_subset = test_images
        image_true_labels = test_labels
    else:
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
    if model_name is not None:
        plt.suptitle(f'Model {model_name} Predictions on Sample Test Data')
    else:
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

def visualize_conf_matrix(model, test_dataset, rep_to_labels, model_name=None):
    """
    Prints classification report and displays confusion matrix

    Parameters
    ----------
    model: model to display results for
    test_dataset: tf.data.Dataset object with test images and labels
    rep_to_labels: dict with keys: integer representation of true labels
                             values: corresponding labels
    model_name: name of model to be displayed in title

    Returns
    -------
    None
    """
    # labels to be displayed
    labels = list(rep_to_labels.values())

    # get predicted and true labels
    y_true = []
    y_pred = []

    for img, label in test_dataset:
        pred = model.predict(img, verbose=0)
        pred_label = rep_to_labels[np.argmax(pred)]
        y_pred.append(pred_label)
        y_true.append(rep_to_labels[int(label)])

    # print classification report
    print(classification_report(y_true, y_pred, labels=labels))

    # plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=[0.5,1.5,2.5,3.5], labels=labels)
    plt.yticks(ticks=[0.5,1.5,2.5,3.5], labels=labels)
    if model_name is not None:
        plt.title(f'Confusion Matrix {model_name}')
    else:
        plt.title('Confusion Matrix')
    plt.show()

from turtle import clear
import tensorflow as tf
from keras import datasets, layers, models, optimizers, utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import os

data_dir = 'data_npy'

def generate_dataset():
    label_dict = {}
    i = 0
    dataset_list = []
    dataset_labels_list = []
    for filename in os.listdir(data_dir):
        if i>15:
            break
        file = os.path.join(data_dir, filename)
        data = np.load(file)
        image_dim = int(math.sqrt(data.shape[1]))
        data_2d = np.reshape(data, (data.shape[0], image_dim, image_dim))
        data_gray = np.expand_dims(data_2d, axis=3)[:250]
        labels = np.full(data_gray.shape[0], i)
        dataset_list.append(data_gray)
        dataset_labels_list.append(labels)
        #labels = utils.to_categorical(labels, len(test_files))
        #X_train, X_test, Y_train, Y_test = train_test_split(data_gray, labels, test_size=0.3, random_state=42)
        label_dict[i] = os.path.splitext(filename)[0]
        i += 1

        #model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test), shuffle=True)
    dataset = np.vstack(tuple(dataset_list))
    dataset_labels = np.concatenate(tuple(dataset_labels_list))
    dataset, dataset_labels = shuffle(dataset, dataset_labels, random_state=0)
    np.save('dataset', dataset)
    np.save('dataset_labels', dataset_labels)
    return label_dict

def create_model():
    file_count = 0
    for path in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, path)):
            file_count += 1

    data_augmentation = models.Sequential([                                    
        layers.RandomFlip(),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),                               
    ])

    model = models.Sequential([
        data_augmentation,
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizers.RMSprop(learning_rate=0.005), metrics=['accuracy'])
    return model

def train_data(model, data_file, label_file):
    dataset = np.load(data_file)
    dataset_labels = np.load(label_file)
    model.fit(dataset, dataset_labels, epochs=10, batch_size=50)
    model.save('model')
    return model
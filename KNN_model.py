"""
Hannah Macdonell x2018zin
ML A1 Assignment Submitted Thursday, October 15th
"""

### I ran isort ###

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sbn
from scipy.stats import mannwhitneyu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def load_num_data(file_path):
    """ Loading 8 and 9 number data into train/test - label/data """
    data = sio.loadmat(file_path)  # loading number data

    # TRAIN: 8 labels and img data
    eights = data["imageArrayTraining8"]  # loading (28,28,750) data shape
    eight_imgs = []
    for j in range(750):
        eight_imgs.append(eights[:, :, j])
        eight_imgs = [x.flatten() for x in eight_imgs]
    eight_labels = [8 for i in range(750)]

    # TRAIN: 9 labels and img data
    nines = data["imageArrayTraining9"]  # loading (28,28,750) data shape
    nine_imgs = []
    for j in range(750):
        nine_imgs.append(nines[:, :, j])
        nine_imgs = [x.flatten() for x in nine_imgs]
    nine_labels = [9 for i in range(750)]

    tr_data = eight_imgs + nine_imgs
    tr_label = eight_labels + nine_labels

    # TEST: 8 labels and img data
    eights = data["imageArrayTesting8"]  # loading (28,28,750) data shape
    eight_imgs = []
    for j in range(250):
        eight_imgs.append(eights[:, :, j])
        eight_imgs = [x.flatten() for x in eight_imgs]
    eight_labels = [8 for i in range(250)]

    # TEST: 9 labels and img data
    nines = data["imageArrayTesting9"]  # loading (28,28,750) data shape
    nine_imgs = []
    for j in range(250):
        nine_imgs.append(nines[:, :, j])
        nine_imgs = [x.flatten() for x in nine_imgs]
    nine_labels = [9 for i in range(250)]

    test_data = eight_imgs + nine_imgs
    test_label = eight_labels + nine_labels

    return tr_data, tr_label, test_data, test_label


def load_mush_data(file_path):
    """ Loading mushroom dataset (8124 samples) into train/test - label/data """

    shroom_df = pd.read_csv(file_path, usecols=list(range(1, 20)), header=None)  # Read in data
    shroom_df = shroom_df.apply(LabelEncoder().fit_transform)  # Encode nominal string data to int

    split = int(len(shroom_df) / 2)  # Split index (8124/2) = 4062

    tr_data = shroom_df[:split].to_numpy()  # first half of csv file, convert to list
    test_data = shroom_df[split:].to_numpy()  # second half, convert to list

    shroom_label = pd.read_csv("data/mushroom.csv", usecols=[0], header=None)  # pull out label classes
    shroom_label = shroom_label.apply(LabelEncoder().fit_transform)  # Encode labels to binary

    # train_test_split
    # full module: sklearn.model_selection.train_test_split
    # pass in full x/y data frame (can specify percentage)
    # output -> tr and test label

    # Sorry flake made this so ugly
    # Dividing labels from shroom_df dataframe and reshaping
    tr_label = (
        shroom_label[:split]
        .to_numpy()
        .reshape(
            split,
        )
    )  # first half of labels
    test_label = (
        shroom_label[split:]
        .to_numpy()
        .reshape(
            split,
        )
    )  # second half of labels

    return tr_data, tr_label, test_data, test_label


def knn_model(tr_data, tr_label, test_data, test_label):
    knn_hits = []  # to store knn prediction hits
    # Running knn model for (1, 20)
    for i in range(1, 21):
        if i == 1:
            i = 30
        knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        knn.fit(tr_data, tr_label)
        hits = 0
        # If knn predicts correctly, count as a knn hit.
        for data, label in zip(test_data, test_label):
            if label == knn.predict([data]):
                hits += 1
        knn_hits.append(hits)

    return knn_hits


def question1(file_path):
    """ Build 20 KNN models with varying K=1,2,3,.....,20 in a loop. """
    tr_data, tr_label, test_data, test_label = load_num_data(file_path)  # "NumberRecognition.mat"
    # List holds number of accurate knn predictions for each
    knn_hits = knn_model(tr_data, tr_label, test_data, test_label)
    knn_error = []
    for i in knn_hits:
        knn_error.append((1 - (i / 500)) * 100)  # list of KNN error values


def question2(file_path):
    """ Calculate AUC values for all features. """
    # Load mushroom data
    tr_data, tr_label, test_data, test_label = load_mush_data(file_path)
    # Join training and testing sets
    x, y = tr_data + test_data, tr_label + test_label
    x = x.transpose()  # Make our feature column data easily accessible as rows
    auc_val = []
    for i in x:
        # This line is from Derek's tutorial
        auc = mannwhitneyu(i, y).statistic / (len(i) * len(y))
        auc_val.append(auc)

    return auc_val


def question3(file_path):
    tr_data, tr_label, test_data, test_label = load_mush_data(file_path)
    print(tr_data[:10])
    # knn_hits = knn_model(tr_data, tr_label, test_data, test_label)
    # knn_error = []
    # for i in knn_hits:
    # knn_error.append((1 - (i / 4062)) * 100)  # list of KNN error values


if __name__ == "__main__":
    # setup / helper function calls here, if using
    question1("data/NumberRecognition.mat")
    question2("data/mushroom.csv")
    question3("data/mushroom.csv")

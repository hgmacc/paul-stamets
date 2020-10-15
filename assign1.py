"""
Hannah Macdonell x2018zin
ML A1 Assignment Submitted Thursday, October 15th
"""


import numpy as np
import scipy.io as sio
import mypy
import isort
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sbn
from pathlib import Path

# numpy.reshape to reshape a 3D matrix of 2D images into a decomposed 2D matrix KNeighborsClassifier for training
# numpy.predict for prediction

from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier
import sklearn


def load_num_data():
    """ Loading number data """
    data = sio.loadmat("NumberRecognition.mat")  # loading number data
    # Setting up X_tr and y_tr
    eights = data["imageArrayTraining8"]  # loading (28,28,750) data shape
    eight_imgs = []
    for j in range(750):
        eight_imgs.append(eights[:, :, j])
        eight_imgs = [x.flatten() for x in eight_imgs]

    eight_labels = [8 for i in range(750)]

    nines = data["imageArrayTraining9"]  # loading (28,28,750) data shape
    nine_imgs = []
    for j in range(750):
        nine_imgs.append(nines[:, :, j])
        nine_imgs = [x.flatten() for x in nine_imgs]
    nine_labels = [9 for i in range(750)]
    tr_data = eight_imgs + nine_imgs
    tr_label = eight_labels + nine_labels

    eights = data["imageArrayTesting8"]  # loading (28,28,750) data shape
    eight_imgs = []
    for j in range(250):
        eight_imgs.append(eights[:, :, j])
        eight_imgs = [x.flatten() for x in eight_imgs]

    eight_labels = [8 for i in range(250)]

    nines = data["imageArrayTesting9"]  # loading (28,28,750) data shape
    nine_imgs = []
    for j in range(250):
        nine_imgs.append(nines[:, :, j])
        nine_imgs = [x.flatten() for x in nine_imgs]
    nine_labels = [9 for i in range(250)]
    test_data = eight_imgs + nine_imgs
    test_label = eight_labels + nine_labels

    return tr_data, tr_label, test_data, test_label

def load_mush_data():
    ''' Loading mushroom dataset (8124 samples) --> remove '''
    data =



    return tr_data, tr_label, test_data, test_label


    data = load_iris(as_frame=True)
    X: DataFrame = data.data  # has shape (150, 4)
    y: DataFrame = data.target  # has shape (150,)
    labels = data.target_names

def knn_model(tr_data, tr_label, test_data, test_label):
    knn_hits = []  # to store knn prediction hits
    for i in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(tr_data, tr_label)
        hits = 0
        for data, label in zip(test_data, test_label):
            if label == knn.predict([data]):
                hits += 1
        knn_hits.append(hits)
    return knn_hits

def plot_knn(knn_hits, graph):
    """ A plot of testing error rate (as a percentage on the y axis) vs. K (x axis). """
    knn_hits = [((1 - (i / 500)) * 100) for i in knn_hits]  # converting knn hit count to % value

    df = pd.DataFrame({"%_Error": knn_hits, "K_Values": list(range(1, 21))})
    sbn.scatterplot(data=df, y="%_Error", x="K_Values", s=100, color=".2", marker="+")
    plt.title(graph)
    plt.show()

# Question 1: Build 20 KNN models with varying K=1,2,3,.....,20 in a loop.
# Provide a plot of testing error rate (as a percentage on the y axis) vs. K (x axis).
# Provide a printout of your code (Matlab or python). Provide a printout of the plot.
# Answer the following questions:
# a) Why does testing error rise at high values of K?
# b) What is the error rate at the lowest K? Do you expect this to be a reliable performance estimate? Why?

def question1():
    """ Build 20 KNN models with varying K=1,2,3,.....,20 in a loop. """
    tr_data, tr_label, test_data, test_label = load_num_data()
    # List holds number of accurate knn predictions for each
    knn_hits = knn_model(tr_data, tr_label, test_data, test_label)
    plot_knn(knn_hits, "Question 1: Error rate (%) vs. K value")

def question2():
    print("Q2")

def question3():
    print("Q3")


if __name__ == "__main__":
    # setup / helper function calls here, if using
    question1()
    question2()  # these functions can optionally take arguments (e.g. `Path`s to your data)
    question3()

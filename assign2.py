"""
Hannah Macdonell x2018zin
ML A2 Assignment Submitted Thursday, October 29th
"""

import typing
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sbn
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_num_data(file_path: str) -> Tuple[np.ndarray]:
    """ Loading numbers into labels/data """

    # Loading .mat data and extracting X/y data to a numpy array
    num_dic: dict = sio.loadmat(file_path)

    # reshaping data to (784, 30000) and (30000,)
    num_imgs: np.ndarray = num_dic["X"].reshape([np.prod(num_dic["X"].shape[:2]), num_dic["X"].shape[-1]])
    num_labels: np.ndarray = num_dic["y"][0]

    # Adding labels to array to pull out all non-8s/9s so shape = (5870, 784)
    num_data: np.ndarray = np.column_stack((num_labels, num_imgs.T))
    num_data = num_data[num_data[:, 0] > 7]

    # Removing labels from data
    num_data = num_data[:, 1:]
    num_labels = num_labels[num_labels > 7]
    print(num_data.shape, num_labels.shape)
    return num_data, num_labels


def load_mush_data(file_path: str) -> Tuple:
    """ Loading mushroom dataset (8124 samples) into labels/data """
    # Switched data to kaggle data set
    shroom_df: pd.DataFrame = pd.read_csv(file_path)  # Read in data
    shroom_df = shroom_df.apply(LabelEncoder().fit_transform)  # Encode nominal string data to int

    data: np.ndarray = shroom_df.drop(["class"], axis=1).values
    labels: np.ndarray = shroom_df["class"].values

    return data, labels


def svm_model(data, labels: Tuple) -> int:
    """Support Vector Machine (linear kernel, RBF kernel where the trainer sets the kernel parameter, gamma/sigma), """

    # Split data into training and testing
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.5)

    # Create and fit support vector model where gamma is auto selected.
    svmodel = SVC(random_state=42, gamma="auto")
    svmodel.fit(train_data, train_label)

    # Return accuracy score
    return svmodel.score(test_data, test_label) * 100


def random_forest_model(data, labels: Tuple) -> int:
    """Random Forest (Number of Trees = 100)"""

    # Split data into training and testing
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.5)

    # Create and fit model with 100 trees
    rf = RF(n_estimators=5, random_state=42)
    rf.fit(test_data, test_label)

    # Return accuracy score
    return rf.score(test_data, test_label) * 100


def knn_model(data, labels: Tuple) -> List[int]:
    """ K-NN (K=1, K=5 and K=10) """
    # Made test size half of data (4062 instances for training and testing)
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.5)

    # Running knn model for K=1, K=5 and K=10
    knn_scores = []
    for i in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        knn.fit(train_data, train_label)
        accuracy = knn.score(test_data, test_label)
        knn_scores.append(accuracy)

    return knn_scores

    # Implement K-Fold cross validation (K=5)
    # Within the validation, you will train and compare the 3 models (SVM, RF, KNN)
    # The validation loop will train these models for predicting 8s and 9s.
    # the exact same set of training data will be used to construct each model being compared to ensure a fair comparison
    # Provide a K Fold validated error rate for each of the classifiers. Provide your code.


def question1(file_path: str) -> None:
    data, labels = load_num_data(file_path)
    # print(knn_model(data, labels))
    # print(random_forest_model(data, labels))
    # print(svm_model(data, labels))
    skf = StratifiedKFold(n_splits=5, random_state=42)

    # scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

    for train_index, test_index in skf.split(data, labels):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


def question2(file_path: str) -> List[int]:
    """ Calculate AUC values for all features. """

    # This is to pull my feature column headers into a list for nice AUC sorting
    headers: list = pd.read_csv(file_path, nrows=1)
    headers = headers.drop(["class"], axis=1).columns.values

    # Load mushroom data, join training and testing sets
    data, labels = load_mush_data(file_path)

    # Make our feature columns -> rows
    data = data.transpose()
    auc_val: list[float] = []

    for row, feat_name in zip(data, headers):
        # This line is from Derek's tutorial
        auc: float = mannwhitneyu(row, labels).statistic / (len(row) * len(labels))
        auc_val.append([feat_name, auc])

    # Sorting AUC values by furthest proximity to 0.50
    auc_val = sorted(auc_val, key=lambda t: abs(0.50 - t[1]), reverse=True)

    # This segment of code is adapted from the assigment submission portal
    FEAT_NAMES = headers
    COLS = ["Feature", "AUC"]
    aucs = pd.DataFrame(
        index=FEAT_NAMES,
        columns=COLS,
        data=(auc_val),
    )

    aucs.to_json(path_or_buf="/Users/hannahmacdonell/PycharmProjects/paul-stamets/aucs.json")

    return auc_val


def question3(file_path: str) -> None:
    data, labels = load_mush_data(file_path)
    pass


if __name__ == "__main__":
    # question1("/Users/hannahmacdonell/PycharmProjects/paul-stamets/data/NumberRecognitionBigger.mat")
    question2("/Users/hannahmacdonell/PycharmProjects/paul-stamets/mushroom.csv")
    # question3("/Users/hannahmacdonell/PycharmProjects/paul-stamets/mushroom.csv")

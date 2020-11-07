"""
Hannah Macdonell x2018zin
ML A3 Assignment Submitted Friday, November 13th
"""

# I ran isort

import statistics
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from keras import losses, optimizers
from keras.layers import Conv2D, Flatten, BatchNormalization
from keras.layers import Dense, ReLU
from keras.models import Sequential

FILEPATH = "dereks-path-here"


def load_num_data(file_path: str) -> Tuple:
    """ Loading numbers into labels/data """

    # Loading .mat data and extracting X/y data to a numpy array
    num_dic: dict = sio.loadmat(file_path)

    # reshaping data to (784, 30000) and (30000,)
    num_data: np.ndarray = num_dic["X"].reshape([np.prod(num_dic["X"].shape[:2]), num_dic["X"].shape[-1]])
    num_labels: np.ndarray = num_dic["y"][0]

    return num_data, num_labels


def load_mush_data(file_path: str) -> Tuple:
    """ Loading mushroom dataset (8124 samples) into labels/data """

    # Switched data to kaggle data set bc other data set was a mess
    shroom_df: pd.DataFrame = pd.read_csv(file_path)  # Read in data
    label_df = shroom_df.apply(LabelEncoder().fit_transform)  # Encode nominal string data to int

    # Dropping 'class' for feature data, pulling out 'class' data for labels
    labels: np.ndarray = label_df["class"].values

    # Using one hot encoding on my mushroom data. I now have 117 features.
    shroom_df = pd.get_dummies(shroom_df.drop(["class"], axis=1))
    data: np.ndarray = shroom_df.values

    return data, labels


def ANN_model(data, labels: np.ndarray) -> pd.DataFrame:
    ann = MLPClassifier(hidden_layer_sizes=5, activation="tanh", solver="lbfgs")
    # Train test split here
    ann.fit(X_train, y_train)
    ann.predict(X_test)


def CNN_model(data, labels: np.ndarray) -> pd.DataFrame:
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv2D(5, kernel_size=3, input_shape=(28, 28, 1), activation="linear", data_format="channels_last"))
    model.add(ReLU())
    model.add(Flatten())
    # multiclass classification output, use softmax
    model.add(Dense(units=10, activation="softmax"))  # 10 units, 10 digits
    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, lr=0.001),
        loss=losses.mean_squared_error,
        metrics=["accuracy"],
    )
    history = model.fit(X_train, y_train, epochs=15, verbose=1)
    y_pred = model.predict(X_test)


def k_fold(data, labels: np.ndarray) -> pd.DataFrame:
    """ This function does k-fold validation using SVM, RF and KNN models. """
    k_scores = []
    # Completing stratified split (k = 5) of data for all models to use
    skf: classmethod = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random forest model
    model = RF(n_estimators=5, random_state=42)
    scores = cross_validate(model, data, labels, scoring="accuracy", cv=skf, n_jobs=-1)
    k_scores.append((1 - statistics.mean(scores["test_score"])))

    # KNN model for 1, 5, and 10 neighbours
    for i in [1, 5, 10]:
        model = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        scores = cross_validate(model, data, labels, scoring="accuracy", cv=skf, n_jobs=-1)
        k_scores.append((1 - statistics.mean(scores["test_score"])))

    err_df = pd.DataFrame([k_scores], columns=["rf", "knn1", "knn5", "knn10"], index=["err"])
    return err_df


def question1(file_path: str) -> pd.DataFrame:
    """ Question 1 performs k_fold on 8/9 number data."""
    data, labels = load_num_data(file_path)


def question2(file_path: str) -> pd.DataFrame:
    """ Calculate AUC values for all mushroom features. """

    # This is to pull my feature column headers into a list for nice AUC sorting
    headers: pd.DataFrame = pd.read_csv(file_path, nrows=1)
    headers = headers.drop(["class"], axis=1).columns.values

    # Load mushroom data, join training and testing sets
    data, labels = load_mush_data(file_path)

    # Make our feature columns -> rows
    data = data.transpose()
    auc_val: list = []

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

    return aucs


def question3(file_path: str) -> pd.DataFrame:
    """ Question 3 performs k_fold on mushroom data."""
    data, labels = load_mush_data(file_path)

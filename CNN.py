"""
Hannah Macdonell x2018zin
ML A3 Assignment Submitted Monday, November 16th
"""

import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from keras import losses, optimizers
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D, ReLU
from keras.models import Sequential
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Completing stratified split (k = 5) of data for all models to use
SKF: Any = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def load_num_data(q) -> Tuple[ndarray, ndarray]:
    """ Loading numbers into labels/data for all models but CNN """

    # Loading .mat data and extracting X/y data to a numpy array
    num_dic: Dict[str, ndarray] = sio.loadmat(NUMFILEPATH)

    if q == "all":
        # reshaping data to (30000, 784) and (30000,)
        num_data: ndarray = num_dic["X"].reshape([np.prod(num_dic["X"].shape[:2]), num_dic["X"].shape[-1]]).T
        num_labels: ndarray = num_dic["y"][0]

    elif q == "CNN":
        # reshaping data to (30000, 784) and (30000,)
        num_data = num_dic["X"].transpose([2, 0, 1]).astype("float32")
        # Adding acis for CNN
        num_data = np.array(num_data)[:, :, :, np.newaxis]
        num_labels = num_dic["y"].flatten()

    elif q == "four":
        # This is for Q4 testing data
        num_dic = sio.loadmat(NUMTESTFILEPATH)
        num_data = num_dic["X"].transpose([2, 0, 1]).astype("float32")
        # Adding axis for CNN
        num_data = np.array(num_data)[:, :, :, np.newaxis]

    else:
        raise TypeError("Q not specified.")

    return num_data, num_labels


def load_mush_data() -> Tuple[ndarray, ndarray]:
    """ Loading mushroom dataset (8124 samples) into labels/data """

    # Switched data to kaggle data set bc other data set was a mess
    shroom_df: pd.DataFrame = pd.read_csv(MUSHFILEPATH)  # Read in data
    label_df: pd.DataFrame = shroom_df.apply(LabelEncoder().fit_transform)  # Encode nominal string data to int

    # Dropping 'class' for feature data, pulling out 'class' data for labels
    labels: ndarray = label_df["class"].values

    # Using one hot encoding on my mushroom data. I now have 117 features.
    shroom_df = pd.get_dummies(shroom_df.drop(["class"], axis=1))
    data: ndarray = shroom_df.values

    return data, labels


def k_fold(data, labels: ndarray) -> pd.DataFrame:
    """This function does k-fold validation using CNN, ANN, RF and KNN models.
    q is used to differentiate between performing k_fold for Q1 and Q2."""

    # Initializing list to hold my kscores
    k_scores: List[int] = []

    # Random forest model
    model: Any = RF(n_estimators=5, random_state=42)
    scores: Any = cross_validate(model, data, labels, scoring="accuracy", cv=SKF, n_jobs=-1)
    k_scores.append((1 - statistics.mean(scores["test_score"])))

    # KNN model for 1, 5, and 10 neighbours
    for i in [1, 5, 10]:
        model = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        scores = cross_validate(model, data, labels, scoring="accuracy", cv=SKF, n_jobs=-1)
        k_scores.append((1 - statistics.mean(scores["test_score"])))

    # My err dataframe only returns random forest and knn scores (used in both Q1 and Q3)
    err_df: pd.DataFrame = pd.DataFrame([k_scores], columns=["rf", "knn1", "knn5", "knn10"], index=["err"])

    return err_df


def cnn_given(data, labels: ndarray) -> Any:
    """ This is the CNN model given in the assignment description. """

    # Splitting data and labels
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42)

    # CNN model for image data from assignment outline
    model: Any = Sequential()
    model.add(BatchNormalization())
    model.add(Conv2D(5, kernel_size=3, input_shape=(28, 28, 1), activation="linear", data_format="channels_last"))
    model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(units=10, activation="softmax"))
    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, lr=0.001), loss=losses.mean_squared_error, metrics=["accuracy"]
    )
    history = model.fit(X_train, y_train, epochs=15, verbose=1)
    # Pulling out accuracy
    score: float = model.evaluate(X_test, y_test, verbose=0)[1]

    return score


def question1() -> pd.DataFrame:
    """ Question 1 performs k_fold on all number data."""

    # Running ANN, KNN and RF
    numbers_temp: Tuple[ndarray, ndarray] = load_num_data("all")
    data: ndarray = numbers_temp[0]
    labels: ndarray = numbers_temp[1]
    # Setting up dataframe holding RF and KNN error
    err_df: pd.DataFrame = k_fold(data, labels)

    # Running given CNN model from outline
    numbers_temp = load_num_data("CNN")
    data = numbers_temp[0]
    labels = numbers_temp[1]
    # adding CNN error to dataframe
    err_df["cnn"] = cnn_given(data, labels)

    return err_df


def question2() -> pd.DataFrame:
    """ Calculate AUC values for all mushroom features. """

    # This is to pull my feature column headers into a list for nice AUC sorting
    headers: pd.DataFrame = pd.read_csv(MUSHFILEPATH, nrows=1)
    headers = headers.drop(["class"], axis=1).columns.values

    # Load mushroom data, join training and testing sets
    mush_data: Tuple[ndarray, ndarray] = load_mush_data()
    data: ndarray = mush_data[0]
    labels: ndarray = mush_data[1]

    # Make our feature columns -> rows
    data = data.transpose()
    auc_val: list = []

    for row, feat_name in zip(data, headers):
        # Calculating auc values with directionality preserved with roc_auc_score method.
        auc: float = roc_auc_score(row, labels)
        auc_val.append([feat_name, auc])

    # Sorting AUC values by furthest proximity to 0.50
    auc_val = sorted(auc_val, key=lambda t: abs(0.50 - t[1]), reverse=True)

    # This segment of code is adapted from the assigment submission portal
    FEAT_NAMES = headers
    COLS = ["Feature", "AUC"]
    aucs = pd.DataFrame(
        index=FEAT_NAMES,
        columns=COLS,
        data=auc_val,
    )

    return aucs


def question3() -> pd.DataFrame:
    """ Question 3 performs k_fold on mushroom data."""
    # Loading in mush data
    mush_data: Tuple[ndarray, ndarray] = load_mush_data()
    data: ndarray = mush_data[0]
    labels: ndarray = mush_data[1]
    err_df: pd.DataFrame = k_fold(data, labels)

    # Basic ANN Model structure given in assignment
    model: Any = MLPClassifier(hidden_layer_sizes=5, activation="tanh", solver="lbfgs")
    scores: Any = cross_validate(model, data, labels, scoring="accuracy", cv=SKF, n_jobs=-1)
    err_df["ann"] = 1 - statistics.mean(scores["test_score"])

    # Testing out ANN models with different hidden layer sizes
    for num in [1, 2, 3]:
        model = MLPClassifier(hidden_layer_sizes=num, activation="tanh", solver="lbfgs")
        scores = cross_validate(model, data, labels, scoring="accuracy", cv=SKF, n_jobs=-1)
        # Adding error scores to k-fold data
        err_df["ann-%s" % num] = 1 - statistics.mean(scores["test_score"])

    # Testing ANNs with different hidden layer activation functions
    for method in [
        "identity",
        "logistic",
        "relu",
    ]:
        model = MLPClassifier(hidden_layer_sizes=1, activation=method, solver="lbfgs")
        scores = cross_validate(model, data, labels, scoring="accuracy", cv=SKF, n_jobs=-1)
        # Adding error scores to k-fold data
        name = "ann-" + method
        err_df[name] = 1 - statistics.mean(scores["test_score"])

    for i in ["lbfgs", "sgd", "adam"]:
        model = MLPClassifier(hidden_layer_sizes=1, activation="tanh", solver=i)
        scores = cross_validate(model, data, labels, scoring="accuracy", cv=SKF, n_jobs=-1)
        # Adding error scores to k-fold data
        name = "ann-" + i
        err_df[name] = 1 - statistics.mean(scores["test_score"])

    return err_df


def question4() -> pd.DataFrame:
    """ This is my optimal cnn design"""
    # Loading in CNN data
    numbers_temp: Tuple[ndarray, ndarray] = load_num_data("CNN")
    data: ndarray = numbers_temp[0]
    labels: ndarray = numbers_temp[1]
    test_data: ndarray = load_num_data("four")

    # CNN model for image data
    model: Any = Sequential()
    model.add(BatchNormalization())
    model.add(Conv2D(5, kernel_size=3, input_shape=(28, 28, 1), activation="linear", data_format="channels_last"))
    model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(units=10, activation="softmax"))
    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, lr=0.001), loss=losses.mean_squared_error, metrics=["accuracy"]
    )

    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, lr=0.001),
        loss=losses.mean_squared_error,
        metrics=["accuracy"],
    )
    history = model.fit(data, labels, epochs=15, verbose=1)
    y_pred = model.predict(test_data)
    return y_pred


if __name__ == "__main__":
    MUSHFILEPATH = str(Path(__file__).resolve().parent / "mushroom.csv")
    NUMFILEPATH = str(Path(__file__).resolve().parent / "data/NumberRecognitionBigger.mat")
    NUMTESTFILEPATH = str(Path(__file__).resolve().parent / "data/NumberRecognitionTesting.mat")
    question1()
    question2()
    question3()
    question4()

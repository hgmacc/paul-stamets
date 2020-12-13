import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import PIL
import scipy.io as sio
import statistics
from keras import losses, optimizers
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D, ReLU
from keras.models import Sequential
from numpy import ndarray
from PIL import Image, ImageFile, ImageOps
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

SKF: Any = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def k_fold(data, labels: ndarray) -> pd.DataFrame:
    """This function does k-fold validation using CNN, ANN, RF and KNN models.
    q is used to differentiate between performing k_fold for Q1 and Q2."""

    # Initializing list to hold my kscores
    k_scores: List[int] = []

    # Random forest model
    model: Any = RF(n_estimators=5, random_state=42)
    scores: Any = cross_validate(
        model, data, labels, scoring="accuracy", cv=SKF, n_jobs=-1
    )
    k_scores.append((1 - statistics.mean(scores["test_score"])))

    # KNN model for 1, 5, and 10 neighbours
    for i in [1, 5, 10]:
        model = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        scores = cross_validate(
            model, data, labels, scoring="accuracy", cv=SKF, n_jobs=-1
        )
        k_scores.append((1 - statistics.mean(scores["test_score"])))

    # My err dataframe only returns random forest and knn scores (used in both Q1 and Q3)
    err_df: pd.DataFrame = pd.DataFrame(
        [k_scores], columns=["rf", "knn1", "knn5", "knn10"], index=["err"]
    )

    return err_df


def cnn_given(data, labels: ndarray) -> Any:
    """ This is the CNN model given in the assignment description. """

    # Splitting data and labels
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42)

    # CNN model for image data from assignment outline
    model: Any = Sequential()
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            5,
            kernel_size=3,
            input_shape=(28, 28, 1),
            activation="linear",
            data_format="channels_last",
        )
    )
    model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(units=10, activation="softmax"))
    model.compile(
        optimizer=optimizers.SGD(momentum=0.9, lr=0.001),
        loss=losses.mean_squared_error,
        metrics=["accuracy"],
    )
    history = model.fit(X_train, y_train, epochs=15, verbose=1)
    # Pulling out accuracy
    score: float = model.evaluate(X_test, y_test, verbose=0)[1]

    return score

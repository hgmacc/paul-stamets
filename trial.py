from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sbn
from scipy.stats import mannwhitneyu
import typing
from typing import Callable, Iterator, Union, Optional, List, Tuple, Dict
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold, StratifiedShuffleSplit


"""
===================================================================================================================
NIFTY DATA TRICKS
===================================================================================================================
"""


shroom_df = pd.read_csv("mushroom.csv")  # Reading in data as data frame

shroom_df.head()  # First five lines of data frame

shroom_df.describe()  # Stats associated with each column/feature including:
#   count - number of instances
#   unique - number of distinct instances i.e. p/e = 2
#   top - most common instance
#   frequency - how many of top

print(shroom_df.shape)  #  Prints (8124, 23) or (instances, features)

shroom_df["class"].unique()  # Prints all unique occurances of your specified feature i.e. 'pe' and 'e'
# Looks like: array(['p','e'], dtype=object)

shroom_df["class"].value_counts()  # Prints counts of all p's and e's within feature 'class'
#   e   4208
#   p   3916
#   Name:  class, dtype: int64

shroom_df = shroom_df.astype("category")
shroom_df.dtypes
# The data is categorical so we’ll use LabelEncoder to convert it to ordinal.
# LabelEncoder converts each value in a column to a number.

labelencoder = LabelEncoder()
for column in shroom_df.columns:
    shroom_df[column] = labelencoder.fit_transform(shroom_df[column])

print(shroom_df.head())
shroom_df = shroom_df.drop(["veil-type"], axis=1)  # This column does not contribute to data (all 0s)

"""
===================================================================================================================
VIOLIN PLOT

A quick look at the characteristics of the data. 

===================================================================================================================
"""

def violinplot():
    df_div = pd.melt(shroom_df, "class", var_name="Characteristics")
    fig, ax = plt.subplots(figsize=(16,6))
    p = sns.violinplot(ax = ax, x="Characteristics", y="value", hue="class", split = True, data=df_div, inner = 'quartile’, palette = 'Set1’)
    df_no_class = shroom_df.drop(["class"],axis = 1)
    p.set_xticklabels(rotation = 90, labels = list(df_no_class.columns));
    plt.savefig("violinplot.png", format='png', dpi=500, bbox_inches='tight') # This is dope

"""
===================================================================================================================
HEAT MAP

Easy way to model correlation between variables. 
Usually, the least correlating variable is the most important one for classification. 

===================================================================================================================
"""
def heatmap():
    plt.figure(figsize=(14,12))
    sns.heatmap(df.corr(),linewidths=.1,cmap="Purples", annot=True, annot_kws={"size": 7})
    plt.yticks(rotation=0);
    plt.savefig("corr.png", format='png', dpi=400, bbox_inches='tight')
    # Output of heat map shows 'gill-colour' is the culprit
    shroom_df[['class', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='class',
                                                                                              ascending=False)

"""
===================================================================================================================
HEAT MAP

Easy way to model correlation between variables. 
Usually, the least correlating variable is the most important one for classification. 

===================================================================================================================
"""

X = shroom_df.drop(['class'], axis=1)
y = shroom_df["class"]X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)


def load_mush_data(file_path):
    """ Loading mushroom dataset (8124 samples) into train/test - label/data """

    """
    ===================================================================================================================
    COMMENT: Everyone who looked up functions to handle encoding their categorical variables got a small bonus for
    this assignment. Congrats!

    BONUS (+2%): Handling Encoding and Normalizing
    ===================================================================================================================
    """
    shroom_df = pd.read_csv(file_path, usecols=list(range(1, 20)), header=None)  # Read in data
    shroom_df = shroom_df.apply(LabelEncoder().fit_transform)  # Encode nominal string data to int

    split = int(len(shroom_df) / 2)  # Split index (8124/2) = 4062

    tr_data = shroom_df[:split].to_numpy()  # first half of csv file, convert to list
    test_data = shroom_df[split:].to_numpy()  # second half, convert to list

    shroom_label = pd.read_csv("mushroom.csv", usecols=[0], header=None)  # pull out label classes
    shroom_label = shroom_label.apply(LabelEncoder().fit_transform)  # Encode labels to binary

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

### FOR A2 ###

def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:
    import numpy as np
    from pathlib import Path
    from pandas import DataFrame

    COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
    if kfold_scores.shape != (1, 6):
        raise ValueError("DataFrame must have 1 row and 6 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

    if np.min(df.values) < 0 or np.max(df.values) > 0.10:
        raise ValueError("Your K-Fold error rates are too extreme. Ensure they are the raw error rates,\r\n"
                         "and NOT percentage error rates. Also ensure your DataFrame contains error rates,\r\n"
                         "and not accuracies. If you are sure you have not made either of the above mistakes,\r\n"
                         "there is probably something else wrong with your code. Contact the TA for help.\r\n")

    if df.loc["err", "svm_linear"] > 0.07:
        raise ValueError("Your svm_linear error rate is too high. There is likely an error in your code.")
    if df.loc["err", "svm_rbf"] > 0.03:
        raise ValueError("Your svm_rbf error rate is too high. There is likely an error in your code.")
    if df.loc["err", "rf"] > 0.05:
        raise ValueError("Your Random Forest error rate is too high. There is likely an error in your code.")
    if df.loc["err", ["knn1", "knn5", "knn10"]].min() > 0.04:
        raise ValueError("One of your KNN error rates is too high. There is likely an error in your code.")

    outfile = Path(__file__).resolve().parent / "kfold_mnist.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")

def save_data_kfold(kfold_scores: pd.DataFrame) -> None:
    import numpy as np
    from pandas import DataFrame
    from pathlib import Path

    COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
    if kfold_scores.shape != (1, 6):
        raise ValueError("DataFrame must have 1 row and 6 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

    outfile = Path(__file__).resolve().parent / "kfold_data.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for individual dataset successfully saved to {outfile}")

# ORIGINAL CODE FOR K-FOLD
    """
        # scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    scores: List[int] = []
    for train_index, test_index in skf.split(data, labels):
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = labels[train_index], labels[test_index]
        model = SVC

        # SVM Model
        svmodel = SVC(random_state=42, gamma="auto")
        svmodel.fit(test_data, test_label)
        svmodel.score(test_data, test_label) * 100

        # Random-forest model
        rf = RF(n_estimators=5, random_state=42)
        rf.fit(test_data, test_label)
        rf.score(test_data, test_label) * 100

        # KNN model with n set to 1, 5, and 10
        knn_scores = []
        for i in [1, 5, 10]:
            knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
            knn.fit(train_data, train_label)
            accuracy = knn.score(test_data, test_label)
            knn_scores.append(accuracy)

        # scores.append()
    """
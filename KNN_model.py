"""
Code Grading Summary
----------------------

| Code Category     | Components     |  Q1   |  Q2   |   Q3    |  Total (/50)  |
| ----------------- | -------------- | :---: | :---: | :-----: | :-----------: |
|                   |                |       |       |         |               |
| Correctness       |                |       |       |         |     27/30     |
|                   |                |       |       |         |               |
|                   | Implementation |  7/8  |  1/2  |   9/10  |     17/20     |
|                   | Results        |  7/7  |  3/3  |         |     10/10     |
|                   |                |       |       |         |               |
| Comprehensibility |                |       |       |         |     15/15     |
|                   |                |       |       |         |               |
|                   | Good Functions |       |       |         |      6/6      |
|                   | Organization   |       |       |         |      5/5      |
|                   | Comments       |       |       |         |      4/4      |
|                   |                |       |       |         |               |
| Readability       |                |       |       |         |     3.5/5     |
|                   |                |       |       |         |               |
|                   | Formatting     |       |       |         |     1.5/2     |
|                   | Linting        |       |       |         |      2/2      |
|                   | No Dead Code   |       |       |         |      0/1      |
|                   |                |       |       |         |               |
| TOTAL             |                |       |       |         |    45.5/50    |

Bonus
-----
Imports: 1%
Encoding/Normalization: 2%
"""

"""
========================================================================================================================
OVERALL COMMENTARY
------------------

You have just the right amount of comments in just the right places. Great! You were clearly able to setup and
properly use the various code tools (flake8, black, isort). However, three lints remained (see automated_report.md),
so you lose a small half mark in Formatting.

Your use of functions was appropriate, and generally you extracted functions that do "just one thing". You've also
Pythonic snake_case names, and the names are clear. I would have extracted largely the same functions myself. Full
marks for functions.

I have removed one implementation point per each of Question 1 and question 2, simply because you should have used
numpy functions and techniques (array programming) rather than native Python loops. Almost everyone lost half a mark
for not showing a solid understanding of the AUC in their code, or 1 mark for not sorting, and that was the same
here.

As there was something very strange about this mushroom data (that zero-valued row) I haven't deducted any other
implementation points for questions 2 and 3. Corrupt data is not your fault, and everything else you did was correct.

So minor numpy issues aside, this is quite straightforward, clean, readable and correct code. Great work!

========================================================================================================================
"""

"""
Hannah Macdonell x2018zin
ML A1 Assignment Submitted Thursday, October 15th
"""
### I ran isort ###

import pandas as pd
import scipy.io as sio
from scipy.stats import mannwhitneyu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


"""
===================================================================================================================
DEDUCTION (-1): Native Python loops when numpy array programming should be used.
===================================================================================================================
"""


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


def knn_model(tr_data, tr_label, test_data, test_label):
    knn_hits = []  # to store knn prediction hits
    # Running knn model for (1, 20)
    for i in [1, 5, 10]:

        knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        knn.fit(tr_data, tr_label)
        hits = 0
        # If knn predicts correctly, count as a knn hit.
        """
        ===============================================================================================================
        COMMENT: You can predict your data all in one step, and get the accuracy via numpy. Just:

        ```
        from sklearn.metrics import accuracy_score
        pred_label = knn.predict(test_data)
        accuracy = accuracy_score(test_label, pred_label)
        ```

        OR, even better, with the one-liner: `accuracy = knn.score(test_data, test_label)`.
        ===============================================================================================================
        """
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
    print(knn_error)


def question2(file_path):
    """ Calculate AUC values for all features. """
    # Load mushroom data
    tr_data, tr_label, test_data, test_label = load_mush_data(file_path)
    # Join training and testing sets
    x, y = tr_data + test_data, tr_label + test_label
    x = x.transpose()  # Make our feature column data easily accessible as rows
    auc_val = []
    """
    ===================================================================================================================
    DEDUCTION (-1): Implementation - AUC Sorting

    There does not appear to be any sorting of the AUC values here. Maybe you did it manually for the table in the
    responses.md file, but manual sorting is really not appropriate for data science or assignment submissions. In
    particular, we want to see that you have some understanding of what the AUC means, and this can only be indicated
    by the proper sorting (which in this case is actualy by distance from 0.5).
    ===================================================================================================================
    """
    for i in x:
        # This line is from Derek's tutorial
        auc = mannwhitneyu(i, y).statistic / (len(i) * len(y))
        auc_val.append(auc)

    return auc_val


def question3(file_path):
    tr_data, tr_label, test_data, test_label = load_mush_data(file_path)
    knn_hits = knn_model(tr_data, tr_label, test_data, test_label)
    knn_error = []
    for i in knn_hits:
        knn_error.append((1 - (i / 4062)) * 100)  # list of KNN error values


for i in range(10):
    print("l")


if __name__ == "__main__":
    # setup / helper function calls here, if using
    question1("/Users/hannahmacdonell/PycharmProjects/paul-stamets/data/NumberRecognition.mat")
    """
    ===================================================================================================================
    DEDUCTION (-1): No Dead Code
    ===================================================================================================================
    """
    # question2("/Users/hannahmacdonell/PycharmProjects/paul-stamets/mushroom.csv")
    # question3("data/mushroom.csv")

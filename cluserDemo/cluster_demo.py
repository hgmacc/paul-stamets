from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn

from numpy import ndarray
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


# This is a central sagittal slice (i.e. sliced between the two hemispheres) of a de-faced brain
# from the NFBS data (http://preprocessed-connectomes-project.org/NFB_skullstripped/)
NFBS_SLICE = Path(__file__).resolve().parent / "NFBS_A00028352.npy"

# For a simple example, let's just demonstrate clustering on the Iris data, since that is naturally
# clustered, and it will be easy to compare the clusters that K-Means finds to the actual clusters.
# This data has four features: 'sepal length', 'sepal width', 'petal length', 'petal width', all in
# centimetres, and 150 samples.
def load_iris_data() -> DataFrame:
    """Load the data and return as a convenient DataFrame for plotting."""
    data = load_iris(as_frame=True)
    X: DataFrame = data.data  # has shape (150, 4)
    y: DataFrame = data.target  # has shape (150,)
    labels = data.target_names
    df = X.copy(deep=True)
    df["target"] = y
    df["label"] = [str(labels[t]) for t in y]
    print(df)
    return df


def plot_iris_kmeans(df: DataFrame) -> None:
    """Plot the clusters found by K-means for various K values"""
    # For now, let's just use the first two features again
    X = df.iloc[:, :2]
    labels = df["label"]
    sbn.set_style("darkgrid")

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        if k == 0:  # plot the True clusters for comparison
            sbn.scatterplot(data=X, x=X.columns[0], y=X.columns[1], hue=labels, ax=ax)
            ax.set_title("True clusters.")
            continue
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        # If the array passed to `hue` looks like a bunch of floats, seaborn will choose a
        # continuos color scheme. We want categorical colors, so we convert to strings.
        categories = np.array(kmeans.labels_, dtype=str)
        sbn.scatterplot(data=X, x=X.columns[0], y=X.columns[1], hue=categories, ax=ax)
        ax.set_title(f"Clusters for K={k}")
    fig.set_size_inches(w=13, h=7)
    plt.draw()
    plt.show(block=False)


def plot_1d_kmeans() -> None:
    """It is often neglected that 1-dimensional K-means is an extremely powerful tool for
    determining natural cutpoints in scalar values. Here, we take the sagittal slice of an MRI,
    flatten it, and apply K-Means to the voxel intensity values. The resulting clusters for small
    values of K are often surprisingly meaningful."""
    img: ndarray = np.load(NFBS_SLICE)
    sbn.set_style("ticks")
    cmap = "inferno"  # this cmap happens to show the clusters distinctly
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        if k == 0:  # plot the True clusters for comparison
            ax.imshow(img, cmap=cmap)
            ax.set_title("Raw MRI image.")
            continue
        kmeans = KMeans(n_clusters=k + 1)
        kmeans.fit(img.reshape(-1, 1))  # 1D K-means needs this shape
        clusters = kmeans.labels_.reshape(img.shape)  # shape back to img shape
        ax.imshow(clusters, cmap=cmap)
        ax.set_title(f"Clusters for K={k+1}")

    for ax in axes.flat:  # tidy up plots for displaying images
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.suptitle("Using K-Means on Pixel Intensity Values")
    fig.set_size_inches(w=13, h=7)
    plt.show(block=False)


def cluster_demo() -> None:
    """Just a helper to ensure plotting displays all at once."""
    plot_iris_kmeans(load_iris_data())
    plot_1d_kmeans()
    plt.show()


if __name__ == "__main__":
    cluster_demo()

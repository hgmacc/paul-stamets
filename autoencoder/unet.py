from pathlib import Path
from typing import Any, Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from numpy import ndarray
from tensorflow import Tensor
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU
from tensorflow.keras.models import Model, Sequential

# Change this variable to `False` to see U-Net results on pictures of cats from
# the PASCAL VOC Segementation dataset. When `True`, will use NFBS MRI brain images.
USE_BRAINS = True

np.random.seed(3)
tf.random.set_seed(3)


def get_portable_path() -> Path:
    """Utility for getting a sensible working directory whether running as a script or in Colab"""
    try:
        outdir = Path(__file__).resolve().parent
        return outdir
    except NameError:
        print("Possible use of Colab detected. Attempting to exploit `globals()`...")
    try:
        outdir = Path(globals()["_dh"][0]).resolve()
        return outdir
    except KeyError:
        print("Colab not detected.")
        print("Defaulting to current working directory for files.")
        return Path().resolve()


if USE_BRAINS:
    DATA = get_portable_path() / "nfbs_brains"
    X_TRAIN = np.expand_dims(np.load(DATA / "X_train.npy"), -1)
    X_TEST = np.expand_dims(np.load(DATA / "X_test.npy"), -1)
    Y_TRAIN = np.expand_dims(np.load(DATA / "Y_train.npy"), -1)
    Y_TEST = np.expand_dims(np.load(DATA / "Y_test.npy"), -1)
    IMG_SIZE = X_TRAIN.shape[1:-1]
    IN_CHANNELS = 1
    IMG_SHAPE = IMG_SIZE + (IN_CHANNELS,)
    N_CLASSES = 1
else:
    DATA = get_portable_path() / "pascal-voc"
    X_TRAIN = np.load(Path(DATA / "cat/train/img/all.npy"))
    X_TEST = np.load(Path(DATA / "cat/test/img/all.npy"))
    Y_TRAIN = np.load(Path(DATA / "cat/train/mask/all.npy"))
    Y_TEST = np.load(Path(DATA / "cat/test/mask/all.npy"))
    # cats have label "8" in the images, so need to be converted to codes
    Y_TRAIN[Y_TRAIN != 8] = 0
    Y_TRAIN[Y_TRAIN == 8] = 1
    Y_TEST[Y_TEST != 8] = 0
    Y_TEST[Y_TEST == 8] = 1
    IMG_SIZE = X_TRAIN.shape[1:-1]
    IN_CHANNELS = 3
    IMG_SHAPE = IMG_SIZE + (IN_CHANNELS,)
    N_CLASSES = 1

BATCH_SIZE = 4
EPOCHS = 20


# The `dice_loss` function below is an adaptation (and correction) of
# Nieradzik, L. (2018, September 27). Loss Functions For Segmentation. Lars76.Github.Io.
# https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html
#
# But see also # https://github.com/pytorch/pytorch/issues/1249
# for PyTorch implementations and hints on how to improve the below implementation, or
# and https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a and
# https://github.com/keras-team/keras/issues/13085
#
# Note also that Dice loss is somewhat specialized that enough that it isn't even
# in a lot of major deep learning libries!
def dice_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    # Note we must subtract one here because the loss is something we want to *minimize*
    return 1 - numerator / denominator


# For monitoring metrics and reporting, we usually want the actual Dice coefficient. That is, we
# want to know if our model get's e.g. 0.98 Dice score.
def dice(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return 1 - dice_loss(y_true, y_pred)


# Implementations below base on above cited plus discussions and code at:
# and https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a and
# https://github.com/keras-team/keras/issues/13085
# But see also https://github.com/pytorch/pytorch/issues/1249 # for discussion of what the
# smoothing factor is. In short, it is supposed to help regularize.
def smooth_dice_loss(smooth: float = 1.0) -> Callable[[Tensor, Tensor], Tensor]:
    def smoothed_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true = tf.cast(y_true, tf.float32)
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth
        return 1 - (2.0 * intersection + smooth) / union

    return smoothed_loss


# For monitoring metrics again, we want the actual dice values.
def smooth_dice(smooth: float = 1.0) -> Callable[[Tensor, Tensor], Tensor]:
    def dice(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return 1 - smooth_dice_loss(smooth)(y_true, y_pred)

    return dice


# The `balanced_cross_entropy` function below adapted from
# Nieradzik, L. (2018, September 27). Loss Functions For Segmentation. Lars76.Github.Io.
# https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html
#
# Adjust beta toward 1 to penalize one kind of misclassification, and towards 0 to
# penalize the opposite misclassification.
def balanced_cross_entropy(beta: float = 0.7) -> Callable[[Tensor, Tensor], Tensor]:
    def bce(y_true: Tensor, y_pred: Tensor) -> Tensor:
        w_a = beta * tf.cast(y_true, tf.float32)
        w_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)
        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (
            w_a + w_b
        ) + y_pred * w_b
        return tf.reduce_mean(o)

    return bce


def plot_results(
    x_test: ndarray, y_test: ndarray, y_pred: ndarray, n_images: int = 10
) -> None:
    """Plots the first `n_images`, the masks, and the learned segmentations"""
    axes: plt.Axes
    _, axes = plt.subplots(nrows=1, ncols=2)
    for i, (brain, pred, mask) in enumerate(zip(x_test, y_pred, y_test)):
        if i == n_images:
            break
        axes[0].imshow(brain, cmap="inferno")
        axes[0].imshow(mask, cmap="Greys", alpha=0.5)
        axes[0].set_title("Actual")
        axes[0].set_axis_off()

        axes[1].imshow(brain, cmap="inferno")
        axes[1].imshow(pred, cmap="Greys", alpha=0.5)
        axes[1].set_title("Predicted")
        axes[1].set_axis_off()
        plt.pause(0.5)
        for ax in axes:
            ax.clear()
    plt.close()


# This implements the key mini-layer / unit of the U-Net, and is implemented based on the
# descriptions included in Figure 2 and the text of:
#
# Cicek, O., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016).
# 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
# ArXiv:1606.06650 [Cs]. http://arxiv.org/abs/1606.06650
#
# This "DoubleConv" layer is in fact a very powerful building block, and can be used in place of a
# regular conv layer in simple classification and/or regression convolutional networks, often with
# excellent results.
class DoubleConv(Model):
    def __init__(self, filters: Tuple[int, int], **input_args: Any):
        super().__init__()
        args1 = {**dict(kernel_size=3), **input_args}
        layers = [
            # We don't include `activation="relu"` because ReLU is done *after*
            # the BatchNorm layer in the above paper
            Conv2D(filters[0], padding="same", **args1),
            BatchNorm(),
            ReLU(),
            Conv2D(filters[1], kernel_size=3, padding="same"),
            BatchNorm(),
            ReLU(),
        ]
        self.model = Sequential(layers)

    def call(self, x: Tensor) -> Tensor:
        return self.model(x)


class ConvUNet(Model):
    def __init__(self) -> None:
        super().__init__()
        input_args = dict(data_format="channels_last", input_shape=IMG_SHAPE)
        # This is our contracting/encoding path
        self.enc1 = DoubleConv((8, 16), **input_args)
        self.pool1 = MaxPool2D(strides=2)
        self.enc2 = DoubleConv((16, 32))
        self.pool2 = MaxPool2D(strides=2)

        # The portion below is given various eclectic names like "join", "merge", "bottom"
        self.bottleneck = DoubleConv((32, 32))

        # This is our expanding/decoding path
        self.up1 = Conv2DTranspose(32, kernel_size=2, strides=2)
        self.dec1 = DoubleConv((16 + 32, 16))
        self.up2 = Conv2DTranspose(16, kernel_size=2, strides=2)
        self.dec2 = DoubleConv((8 + 16, 8))

        # The output layer in a U-Net is almost always a 1x1 convolution. Depending on how the
        # losses are calculated, activation may be "signmoid" or "softmax", or it may be "linear"
        # (no activation) for certain implementations.
        self.out = Conv2D(N_CLASSES, kernel_size=1, activation="sigmoid")

    def call(self, x: Tensor) -> Tensor:
        # Normally, this code would have more comments and better variable names, but we don't want
        # to make the answers to the U-Net question *too* obvious. Thus the variable names are very
        # mildly obfuscated / poor.
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        x = self.bottleneck(x)

        up1 = self.up1(x)
        r1 = tf.concat([enc2, up1], axis=-1)
        dec1 = self.dec1(r1)
        up2 = self.up2(dec1)
        r2 = tf.concat([enc1, up2], axis=-1)
        dec2 = self.dec2(r2)
        x = self.out(dec2)
        return x


def test_unet(loss: Union[str, Callable] = None) -> None:
    model = ConvUNet()
    model.build((None,) + IMG_SHAPE)
    model.summary()

    # This ugly blog just handles some very boring options logic. Feel free to skip it
    kwargs = dict(optimizer="rmsprop", metrics=["acc", dice])
    if loss == "dice" or (callable(loss) and loss.__name__ == "dice_loss"):
        model.compile(loss=dice_loss, **kwargs)
    elif loss == "smooth_dice":
        model.compile(loss=smooth_dice_loss(), **kwargs)
    elif callable(loss) and loss.__name__ == "smoothed_loss":
        model.compile(loss=loss, **kwargs)
    elif callable(loss) and loss.__name__ == "bce":
        model.compile(loss=loss, **kwargs)
        print("Using balanced_cross_entropy loss.")
    elif callable(loss):
        model.compile(loss=loss, **kwargs)
    else:
        model.compile(loss="binary_crossentropy", **kwargs)

    # NOTE: we use a small batch size here because it is very easy to blow GPU memory with U-Nets,
    # even when they are shallow like this one. You may be able to speed up training by increasing
    # `batch_size` below, but you might also get errors related to GPU memory. If you do, try
    # decreasing the batch size to even 1 or 2.
    #
    # Another reason we use a smaller batch size is that larger batches actually do not work very
    # well in this case (which you can verify yourself if you are interested).
    model.fit(
        X_TRAIN,
        Y_TRAIN,
        validation_data=(X_TEST, Y_TEST),
        batch_size=BATCH_SIZE,
        workers=4,
        epochs=EPOCHS,
    )
    Y_PRED = model.predict(X_TEST)

    plot_results(X_TEST, Y_TEST, Y_PRED, 10)


if __name__ == "__main__":
    test_unet()
    # test_unet(dice_loss)
    test_unet(smooth_dice_loss(2))
    test_unet(balanced_cross_entropy(1))

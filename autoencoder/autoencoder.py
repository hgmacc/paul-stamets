import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow import Tensor
from tensorflow.keras.datasets.fashion_mnist import load_data as load_fashion_mnist
from tensorflow.keras.datasets.mnist import load_data as load_mnist
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.utils.layer_utils import count_params


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


np.random.seed(3)
tf.random.set_seed(3)

MNIST = True
OUTDIR = get_portable_path() / ("mnist_outputs" if MNIST else "fashion_mnist_output")
os.makedirs(OUTDIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (28, 28)
IN_CHANNELS = 1
IMG_SHAPE = IMG_SIZE + (IN_CHANNELS,)
FULL_SHAPE = (BATCH_SIZE,) + IMG_SHAPE
NOISE = 0.2

N_PLOT = 10
SADFACE = (
    np.load(OUTDIR.parent / "sadface.npy").astype("float32").reshape([1, *IMG_SIZE, 1])
    / 255
)

MNIST_PATH = get_portable_path() / "mnist.npz"
(X_train, y_train), (X_test, y_test) = load_mnist(path=str(MNIST_PATH))
X_train = np.expand_dims(X_train, 3).astype("float32") / 255
X_test = np.expand_dims(X_test, 3).astype("float32") / 255


def code_to_image(code: ndarray) -> ndarray:
    """Pads the flat vector to the nearest square size, and then reshapes to a square image. This
    is solely to allow visualizing the linear encoded feature map as a convenient image.

    Parameters
    ----------
    code: ndarray
        The flat (1D) array to be padded and converted to the nearest square-array.
    """
    sq = np.sqrt(code.shape[0])
    w = int(np.ceil(sq))
    padded = np.pad(code, pad_width=(0, w * w - len(code)))
    img = padded.reshape(w, w)
    return img


def plot_results(
    X_test: ndarray,
    encoded: ndarray,
    decoded: ndarray,
    noised: ndarray = None,
    title: str = "",
    save: str = "plot.png",
) -> None:
    """Plot and compare test inputs, encoded representations, and decoded images (and noised images, if used)"""
    noise = isinstance(noised, Tensor) or isinstance(noised, ndarray)
    arrays = [X_test, encoded, decoded]
    subtitles = ["original", "encoded", "decoded"]
    if noise:
        arrays.insert(1, noised)
        subtitles.insert(1, "corrupted")

    fig: plt.Figure
    axes: plt.Axes
    fig, axes = plt.subplots(nrows=len(arrays), ncols=N_PLOT + 1)  # +1 for sadface
    fig.set_size_inches(22, 2 * len(arrays))
    for i in range(N_PLOT + 1):
        for row, (imgs, subtitle) in enumerate(zip(arrays, subtitles)):
            img = imgs[i]
            if subtitle == "encoded" and len(img.shape) == 1:
                img = code_to_image(img)
            axes[row, i].imshow(
                img.squeeze(), cmap="Greys"
            )  # squeeze for Colab bug / older Matplotlib issue
            axes[row, i].set_title(subtitle)
            axes[row, i].set_axis_off()
    fig.suptitle(title, fontsize=20)
    # NOTE: Currently this just saves the plots with descriptive filenames. On Colab, this *may*
    # cause issues, and you might instead want to comment out the `savefig` and `plt.close()` calls
    # and instead use the `plt.show()` and/or `plt.pause()` lines
    fig.savefig(OUTDIR / save)
    print(f"Saved plot to {OUTDIR / save}.")
    # plt.show()
    # plt.pause(3)
    plt.close()


# Here, we are subclassing the `Model` class. This is a good way to build your own models when your
# network design is a little bit more complicated than a `Sequential` model, or when you want to
# access things that you wouldn't really be able to access with such a model. For example, here,
# we ultimately want to be able to access the encoder and decoder portions separately. This would
# not be possibly with a Sequential model. See https://www.tensorflow.org/tutorials/generative/autoencoder
# for a simpler example of this.
class LinearAutoencoder(Model):
    """Implements an undercomplete linear autoencoder. The autoencoder is undercomplete so long as
    `code_dim` is less than the size of the image inputs, because the input is forcibly reduced to
    representation of size `code_dim`.

    Parameters
    ----------
    depth: int
        How many layers (approximately) in the encoding and decoding portions, each. Does not count
        batch norm layers.
    layer_size: int
        The "widith" of each linear layer, e.g. how many units are included.
    code_dim: int
        The size (length) of the hidden representation.
    bnorm: bool
        If True, add in BatchNorm layers to help with training and generalization.
    """

    def __init__(self, depth: int, layer_size: int, code_dim: int, bnorm: bool) -> None:
        super().__init__()
        L = layer_size

        # when defining a model, if you pass an `input_shape` argument in to the first layer,
        # Tensorflow can automatically compile and check for you that you don't have shape errors
        self.encoder = Sequential([Flatten(input_shape=IMG_SHAPE)])
        for _ in range(depth):
            self.encoder.add(Dense(L, activation="relu"))
            if bnorm:
                self.encoder.add(BatchNorm())
        self.encoder.add(Dense(code_dim, activation="relu"))

        # This defines another internal Sequential model, so we pass in an `input_shape` argument
        # again for the first layer
        self.decoder = Sequential(Dense(L, activation="relu", input_shape=(code_dim,)))
        if bnorm:
            self.encoder.add(BatchNorm())
        for _ in range(depth - 1):
            self.decoder.add(Dense(L, activation="relu"))
            if bnorm:
                self.encoder.add(BatchNorm())
        # Our final activation is linear (i.e. no activation). Why? One hint is that since out loss
        # is MSQE, we are doing regression. For regression, you want to allow a full range of
        # outputs, and probably don't want to threshold values.
        #
        # However, another reason is that if you change this final activation to e.g. "relu",
        # you'll discover the decoded outputs have "freckles" or "dead pixels" sprinkled throughout.
        # These are pixels from the incoming feature map that a linear activation *would* have been
        # able to map to a meaningful value, but which the ReLU sent to zero.
        self.decoder.add(Dense(np.prod(IMG_SHAPE), activation="linear"))
        self.decoder.add(Reshape(IMG_SHAPE))

    # When defining a custom model, you have to tell Keras how to pass data through it. That is, we
    # have to define the computational graph. Here, we tell Keras that data must first pass through
    # the encoder, and then that encoded representation must be passed through the decoder. Note
    # our types are Tensors, because the graph is defined by how Tensors are passed along.
    def call(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Here, we build a naive (inefficient) convolutional autoencoder by simply stacking convolutions
# in the encoding path, and then stacking the appropriate number of transposed convolutions in the
# decoding path. More efficient architectures would employ strided convolutions and/or MaxPooling
# layers, but this make determining the number and size of the transposed convolutions *much* more
# difficult, especially if you want to easily be able to adjust depth. So we settle on the simple
# architecture below.
class ConvAutoencoder(Model):
    def __init__(self, depth: int = 8) -> None:
        super().__init__()
        conv_args = dict(
            filters=4, kernel_size=3, activation="relu"
        )  # save space / repetition

        # Build an encoding path
        self.encoder = Sequential()
        self.encoder.add(
            Conv2D(data_format="channels_last", input_shape=IMG_SHAPE, **conv_args)
        )
        self.encoder.add(BatchNorm())
        for _ in range(depth - 1):
            self.encoder.add(Conv2D(**conv_args))
            self.encoder.add(BatchNorm())

        # this line also forces a bottleneck of sorts, in that it forces the code to have 1 channel
        # this is why the encoded representation can be plotted as a black and white images. Note
        # that if you changed below to `filters=3`, then you could see what coloured feature maps
        # would look like.
        self.encoder.add(Conv2D(padding="same", **conv_args))

        # Build a decoding path
        encodeshape = self.encoder.output_shape[1:]
        self.decoder = Sequential()
        self.decoder.add(
            Conv2DTranspose(padding="same", input_shape=encodeshape, **conv_args)
        )
        self.encoder.add(BatchNorm())
        for _ in range(depth - 1):
            self.decoder.add(Conv2DTranspose(**conv_args))
            self.encoder.add(BatchNorm())
        self.decoder.add(Conv2DTranspose(filters=1, kernel_size=3, activation="linear"))

    def call(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def test_linear_autoencoder(
    depth: int, layer_size: int, code_dim: int, bnorm: bool
) -> Sequential:
    # we try our best to seed whatever we can, but the reality is that different GPUs are not always
    # guaranteed to perform calculations the same, and there are other subtleties about Tensorflow
    # that may not ensure 100% reproducible results. However, this does at least seem to ensure similar
    # initial weights and general overall results moreso than just using unseeded code.
    #
    # NOTE: I would *not* actually seed a model if I were developing a real model, since dependency
    # on a seed generally would indicate something wrong about the model. The variation from run to
    # run is actually a good thing for understanding generalization and your model's performance.
    np.random.seed(3)
    tf.random.set_seed(3)
    model = LinearAutoencoder(depth, layer_size, code_dim, bnorm)
    model.build(input_shape=FULL_SHAPE)
    model.summary()
    # autoencoders use "msqe" loss most of the time, or some other regression loss
    model.compile(optimizer="adam", loss=MeanSquaredError(name="msqe"))
    history = model.fit(
        X_train,
        X_train,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(X_test, X_test),
        batch_size=BATCH,
        workers=4,
    )

    # here, we just conveniently sneak some phoney input into our testing samples to help us
    # evaluate what are models are *really* learning
    test_samples = np.concatenate([X_test[:N_PLOT], SADFACE])
    encoded = model.encoder(test_samples).numpy()
    decoded = model.decoder(encoded).numpy()
    loss = np.round(history.history["val_loss"][-1], 3)

    params = count_params(model.trainable_weights)
    title = (
        f"Linear autoencoder with {depth} size-{layer_size} layers ({params} trainable params). "
        f"Code dim={code_dim} val_msqe={loss}"
    )
    save = "linear-autoenc_depth{:02d}_code-dim{:03d}_layer-size{:03d}".format(
        depth, code_dim, layer_size
    )
    e = "_{:02d}epochs".format(EPOCHS) if bnorm else ""
    save += f"{'_bnorm' if bnorm else ''}{e}.png"
    plot_results(test_samples, encoded, decoded, None, title, save)


def test_denoising_linear_autoencoder(
    depth: int, layer_size: int, code_dim: int, bnorm: bool
) -> Sequential:
    np.random.seed(3)
    tf.random.set_seed(3)
    # The correct way to implement a denoising autoencoder would actually be to inject new random noise
    # on each epoch for each sample with an augmentation pipeline. However, the standard pipelines in
    # Keras do not include any nice, convenient ways to add noise, so instead we just create a single
    # corrupted copy of the training data once beforehand.
    X_train_corrupted = tf.constant(X_train) + tf.random.normal(
        X_train.shape, mean=0, stddev=NOISE
    )
    model = LinearAutoencoder(depth, layer_size, code_dim, bnorm)
    model.build(input_shape=FULL_SHAPE)
    model.summary()
    model.compile(optimizer="adam", loss=MeanSquaredError(name="msqe"))
    history = model.fit(
        X_train,
        X_train_corrupted,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(X_test, X_test),
        batch_size=BATCH,
        workers=4,
    )

    test_samples = np.concatenate([X_test[:N_PLOT], SADFACE])
    X_test_corrupted = tf.constant(test_samples) + tf.random.normal(
        test_samples.shape, mean=0, stddev=NOISE
    )
    encoded = model.encoder(test_samples).numpy()
    decoded = model.decoder(encoded).numpy()
    loss = np.round(history.history["val_loss"][-1], 3)

    params = count_params(model.trainable_weights)
    title = (
        f"Linear denoising autoencoder with {depth} size-{layer_size} layers "
        f"({params} trainable params). Code dim={code_dim} val_msqe={loss}"
    )
    save = (
        "linear-denoise-autoenc_depth{:02d}_code-dim{:03d}_layer-size{:03d}.png".format(
            depth, code_dim, layer_size
        )
    )
    plot_results(test_samples, encoded, decoded, X_test_corrupted, title, save)


def test_fullyconv_autoencoder(depth: int) -> Sequential:
    np.random.seed(2)
    tf.random.set_seed(2)
    model = ConvAutoencoder(depth=depth)
    model.build(input_shape=FULL_SHAPE)
    model.summary()
    model.compile(optimizer="adam", loss=MeanSquaredError(name="msqe"))
    history = model.fit(
        X_train,
        X_train,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(X_test, X_test),
        batch_size=BATCH,
        workers=4,
    )

    test_samples = np.concatenate([X_test[:N_PLOT], SADFACE])
    encoded = model.encoder(test_samples).numpy()
    decoded = model.decoder(encoded).numpy()
    loss = np.round(history.history["val_loss"][-1], 3)

    h = encoded[0].shape[0]
    w = encoded[0].shape[1]

    params = count_params(model.trainable_weights)
    e = "_{:02d}epochs".format(EPOCHS)
    save = "fcn-autoenc_depth{:02d}".format(depth)
    save += f"{e}.png"
    title = (
        f"Fully-conv autoencoder with {depth} conv layers ({params} trainable params). "
        f"Code dim={h}x{w} ({h*w}) val_msqe={loss}"
    )
    plot_results(test_samples, encoded, decoded, None, title, save)


def test_denoising_conv_autoencoder(depth: int) -> Sequential:
    np.random.seed(2)
    tf.random.set_seed(2)
    X_train_corrupted = tf.constant(X_train) + tf.random.normal(
        X_train.shape, mean=0, stddev=NOISE
    )
    model = ConvAutoencoder(depth=depth)
    model.build(input_shape=FULL_SHAPE)
    model.summary()
    model.compile(optimizer=RMSprop(), loss=MeanSquaredError(name="msqe"))
    history = model.fit(
        X_train,
        X_train_corrupted,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(X_test, X_test),
        batch_size=BATCH,
        workers=4,
    )

    test_samples = np.concatenate([X_test[:N_PLOT], SADFACE])
    X_test_corrupted = tf.constant(test_samples) + tf.random.normal(
        test_samples.shape, mean=0, stddev=NOISE
    )
    encoded = model.encoder(test_samples).numpy()
    decoded = model.decoder(encoded).numpy()
    loss = np.round(history.history["val_loss"][-1], 3)

    h = encoded[0].shape[0]
    w = encoded[0].shape[1]

    params = count_params(model.trainable_weights)
    title = (
        f"Denoising conv autoencoder with {depth} layers ({params} trainable params). "
        f"Noise={NOISE} Code dim={h}x{w} ({h*w}) val_msqe={loss}"
    )
    save = "fcn-denoise-autoenc_depth{:02d}.png".format(depth)
    plot_results(test_samples, encoded, decoded, X_test_corrupted, title, save)


EPOCHS = 2
BATCH = 32

if __name__ == "__main__":
    """
    for depth in [1, 4, 8, 12]:
        for code_dim in [16, 64, 256]:
            for L in [64, 128, 256]:
                test_linear_autoencoder(
                    depth, layer_size=L, code_dim=code_dim, bnorm=False
                )
    """
    try:  # Compare how BatchNorm improves training of deeper networks
        EPOCHS = 10
        test_linear_autoencoder(depth=12, layer_size=256, code_dim=256, bnorm=False)
        EPOCHS = 10
        test_linear_autoencoder(depth=12, layer_size=256, code_dim=256, bnorm=True)
        EPOCHS = 50
        test_linear_autoencoder(depth=12, layer_size=256, code_dim=256, bnorm=True)
    finally:
        EPOCHS = 2  # ensure we reset the global that we DEFINITELY SHOULD NOT be using

    try:  # the convolutional networks benefit more from heavier training
        EPOCHS = 50
        for depth in [4, 6, 8, 10, 12]:
            test_fullyconv_autoencoder(depth=depth)
    finally:
        EPOCHS = 2  # ensure we reset the global we should NOT be using

    # NOTE: the lines below were not covered in the assignment, but if you want to see what the
    # results for denoising autoencoders look like, you can follow through the code.
    for depth in [1, 4, 8, 12]:
        for code_dim in [16, 64, 256]:
            for L in [64, 128, 256]:
                test_denoising_linear_autoencoder(
                    depth, layer_size=L, code_dim=code_dim, bnorm=False
                )

    for depth in [4, 8, 12]:
        test_denoising_conv_autoencoder(depth)

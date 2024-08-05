import tensorflow_datasets as tfds
import tensorflow as tf


def get_dataset():
    """
    Download the dataset
    """
    BUFFER_SIZE = 10000

    # Scale the MNIST data from [0, 255] range to [0, 1] range
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    # Download the fashion mnist dataset
    datasets, info = tfds.load(name="fashion_mnist", with_info=True, as_supervised=True)
    train = datasets["train"]

    return train.map(scale).cache().shuffle(BUFFER_SIZE)

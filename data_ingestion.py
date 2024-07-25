import tensorflow_datasets as tfds
import tensorflow as tf


def get_dataset():
    """
    Download training dataset
    """
    BUFFER_SIZE = 10000

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, info = tfds.load(name="cifar10", with_info=True, as_supervised=True)
    train = datasets["train"]
    return train.map(scale).cache().shuffle(BUFFER_SIZE)

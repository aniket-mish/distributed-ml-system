import tensorflow_datasets as tfds


def get_dataset():
    """
    Get the data
    """
    BUFFER_SIZE = 10000

    # Scale the mnist data from [0, 255] range to [0, 1] range
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    # Use Fashion-MNIST: https://www.tensorflow.org/datasets/catalog/fashion_mnist
    datasets, info = tfds.load(name="fashion_mnist", with_info=True, as_supervised=True)

    train = datasets["train"]

    return train.map(scale).cache().shuffle(BUFFER_SIZE)

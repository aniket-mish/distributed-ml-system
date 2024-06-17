import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

model = keras.models.load_model("trained_model/saved_model_versions/1")

# Scaling mnist data from (0, 255] to (0., 1.]
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

datasets, info = tfds.load(name="fashion_mnist", with_info=True, as_supervised=True)

ds = datasets["test"].map(scale).cache().shuffle(10000).batch(64)

# Evaluation
test_loss, test_acc = model.evaluate(ds)

# Print
print(f"Test loss: {test_loss} and Test accuracy: {test_acc}")

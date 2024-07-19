import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import shutil
import os


# Scale the data from range [0, 255] to range [0, 1]
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


# Variable to track the best model
best_model_path = ""
best_accuracy = 0


for i in range(1, 4):
    model_path = "trained_model/saved_model_versions/" + str(i)
    model = keras.models.load_model(model_path)
    datasets, info = tfds.load(name="fashion_mnist", with_info=True, as_supervised=True)
    mnist_test = datasets["test"]
    ds = mnist_test.map(scale).cache().shuffle(10000).batch(64)
    loss, accuracy = model.evaluate(ds)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = model_path

destination = "trained_model/saved_model_versions/4"
if os.path.exists(destination):
    shutil.rmtree(destination)

shutil.copytree(best_model_path, destination)
print(f"Best model with accuracy {best_accuracy} is copied to {destination}")

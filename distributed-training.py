import argparse
import json
import os
import tensorflow as tf

from data_ingestion import get_dataset

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Learning rate decay
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


def build_and_compile_cnn_model():
    print("Training CNN model")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1), name="image_bytes"))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.summary()

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def build_and_compile_cnn_model_with_batch_norm():
    print("Training CNN model with batch normalization")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1), name="image_bytes"))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("sigmoid"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("sigmoid"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.summary()

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def build_and_compile_cnn_model_with_dropout():
    print("Training CNN model with dropout")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1), name="image_bytes"))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.summary()

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def main(args):

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    with strategy.scope():
        dataset = get_dataset().batch(BATCH_SIZE).repeat()

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        dataset = dataset.with_options(options)

        model_type = args.model_type

        if model_type == "cnn":
            multi_worker_model = build_and_compile_cnn_model()
        elif model_type == "cnn_batchnorm":
            multi_worker_model = build_and_compile_cnn_model_with_batch_norm()
        elif model_type == "cnn_dropout":
            multi_worker_model = build_and_compile_cnn_model_with_dropout()
        else:
            Exception(f"Entered {model_type} is not supported")

    def _preprocess(bytes_inputs):
        decoded = tf.io.decode_jpeg(bytes_inputs, channels=1)
        resized = tf.image.resize(decoded, size=(28, 28))
        return tf.cast(resized, dtype=tf.uint8)

    def _get_serve_image_fn(model):
        @tf.function(
            input_signature=[tf.TensorSpec([None], dtype=tf.string, name="image_bytes")]
        )
        def serve_image_fn(bytes_inputs):
            decoded_images = tf.map_fn(_preprocess, bytes_inputs, dtype=tf.uint8)
            return model(decoded_images)

        return serve_image_fn

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = args.checkpoint_dir

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(
                "\nLearning rate for epoch {} is {}".format(
                    epoch + 1, multi_worker_model.optimizer.lr.numpy()
                )
            )

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True
        ),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR(),
    ]

    multi_worker_model.fit(dataset, epochs=1, steps_per_epoch=70, callbacks=callbacks)

    def _is_chief():
        return TASK_INDEX == 0

    if _is_chief():
        model_path = args.saved_model_dir

    else:
        # Save to a path that is unique across workers.
        model_path = args.saved_model_dir + "/worker_tmp_" + str(TASK_INDEX)

    multi_worker_model.save(model_path)

    signatures = {
        "serving_default": _get_serve_image_fn(
            multi_worker_model
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="image_bytes")
        )
    }

    tf.saved_model.save(multi_worker_model, model_path, signatures=signatures)


if __name__ == "__main__":
    tf_config = json.loads(os.environ.get("TF_CONFIG") or "{}")
    TASK_INDEX = tf_config["task"]["index"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saved_model_dir", type=str, required=True, help="Tensorflow export directory"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Tensorflow checkpoint directory",
    )

    parser.add_argument("--model_type", type=str, required=True, help="Model type")

    parsed_args = parser.parse_args()
    main(parsed_args)

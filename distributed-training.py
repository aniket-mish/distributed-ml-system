import argparse
import json
import os

import tensorflow as tf
from data_ingestion import get_dataset


def decay(epoch):
    """
    Learning rate decay
    """
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


def build_and_compile_cnn_model():
    """
    Neural net
    """
    print("Training a simple neural net")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(3, 32, 32)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def main(args):
    """
    Distributed training
    """
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
            model = build_and_compile_cnn_model()
        else:
            Exception(f"Entered {model_type} is not supported")

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = args.checkpoint_dir

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(
                "\nLearning rate for epoch {} is {}".format(
                    epoch + 1, model.optimizer.lr.numpy()
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

    model.fit(dataset, epochs=5, steps_per_epoch=70, callbacks=callbacks)

    def _is_chief():
        return TASK_INDEX == 0  # Returns true if master

    # Check if master/worker node
    if _is_chief():
        model_path = args.saved_model_dir
    else:
        # Save to a path that is unique across workers
        model_path = args.saved_model_dir + "/worker_tmp_" + str(TASK_INDEX)

    # Save model
    model.save(model_path)
    tf.saved_model.save(
        model, model_path
    )  # Returns a object that has functions required for inference


if __name__ == "__main__":
    """
    Entrypoint
    """

    tf_config = json.loads(os.environ.get("TF_CONFIG") or "{}")
    TASK_INDEX = tf_config["task"]["index"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saved_model_dir", type=str, required=True, help="tf export directory"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="tf checkpoint directory",
    )

    parser.add_argument("--model_type", type=str, required=True, help="model type")

    parsed_args = parser.parse_args()

    main(parsed_args)

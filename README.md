# Distributed Machine Learning System

Building a distributed ML system in this modern era of deep learning is a necessity. Every company that uses ML wants to serve their customers at scale. Models are becoming huge and the datasets required to train these models are increasing as well. See [chinchilla scaling laws](https://arxiv.org/abs/2203.15556). On top of that GPUs are expensive. So keeping these GPUs idle can cost you a lot. Using multi-GPU training and optimizing inference can save costs and make user experience good.

I am working on this project to get deeper understanding of distributed ML systems, use kubernetes, tensorflow and argo workflows.

## Background

> Distributed systems are a group of nodes that talk to each other to achieve a specific task, such as streaming movies across devices, search engines, etc. - Understanding distributed systems

These systems handle massive amounts of data across multiple clusters, use automation tools, and benefit from hardware accelerations.

This repository includes code and references to implement a scalable and reliable machine learning system. I'm constructing all the components including data ingestion, training, serving, and monitoring these workloads.

I'm building an image classification end-to-end system.

The steps involved are:
1. Setup
2. Data Ingestion
3. Distributed Training
4. Evaluation
5. Serving
6. End-to-End Workflow

## Setup

I'm using a Macbook. These system are generally built on cloud. I'm using conda as the package manager. I also use homebrew for installations.

[1] Let's install `Tensorflow` for data processing, model building and evaluation workflows.

```bash
conda install Tensorflow
```

[2] `Docker` is required to create single- or multi-node clusters. I'm installing docker desktop.

[3] Install a popular CLI utility called `kubectl`.

```bash
brew install kubectl
```

[4] To use Kubernetes on the local machine, install [k3d](https://k3d.io/v5.5.2/), a lightweight wrapper to run K8s. There's minikube, kind and other distributions as well but I find k3d lean, memory efficient and simpler.

```bash
wget -q -O - https://raw.githubusercontent.com/rancher/k3d/main/install.sh | bash
```

Create a single-node cluster. You can create a multi-server cluster as well by specifying `--servers 3`.

```bash
k3d cluster create fmnist --image rancher/k3s:v1.27.12-k3s1
```

You can see the cluster info using command

```bash
kubectl cluster-info
```

Let's see which pods are created using

```bash
kubectl get nodes
```

[5] Install `kubectx` to easily switch between clusters and `kubens` for namespaces. This is a very handy utility.

```bash
brew install kubectx
```

Using kubens you can swtich between namespaces easily

#TODO

[6] Next, we install kubeflow training operator that allows us to train large models effectively.

But first create a new namespace.

```bash
kubectl create namespace kubeflow

kubens kubeflow
```

Now install the operator.

```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
```

[7] To create an end-to-end workflow we need argo workflows.

```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.7/install.yaml
```

[8] For experiment tracking, install MLFlow.

#TODO

## System Architecture

There are multiple design patterns which can be used to create a ML system. In this project, I'm sticking to the easiest one. It has a data ingestion component. Once data is available you can schedule the pipeline to download the data and store it somewhere(e.g. s3). We then train multiple models on the same dataset parallely. Once the models are available, we can pick the best model and create a scalable inference service. 

<img width="1075" alt="Screenshot 2024-06-17 at 3 28 42 PM" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/635143bb-0952-4578-99cd-6d40d1172a33">

## Data Ingestion

I'm using the fashion mnist dataset that has 70,000 images(60,000 for training and 10,000 for evaluation). It has 10 different catergories and each image has a low resolution of 28x28 px.

#TODO

### Create a simple pipeline

The `tf.data` API enables you to build complex input pipelines from simple, reusable pieces. It's very efficient and enables handling large amounts of data, reading from different data formats, and performing complex transformations.

I'm loading the dataset into a `tf.data.Dataset` object and cast the images to float32. Next, I'm normalizing the image pixel values from the [0, 255] to the [0, 1] range. These are some standard practices. I'm keeping an *in-memory cache* to improve performance. Let's also shuffle the training data to add some randomness.

```python
import Tensorflow_datasets as tfds
import Tensorflow as tf

def get_dataset():
    BUFFER_SIZE = 10000
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    train = datasets['train']
    return train.map(scale).cache().shuffle(BUFFER_SIZE)
```

I'm using Tensorflow datasets module to load the dataset. The above piece of code gives a shuffled dataset where each element consists of images and labels.

### Create a distributed data pipeline

To consume a large dataset(>PBs), we need to use a distributed approach. We can do that with some tweaks to the same function that we created. 

For distributed data ingestion, just increase the batch size to use the extra computing power effectively, 

> [!TIP]
> Use the largest batch size that fits the GPU memory

There are several strategies in-built into Tensorflow library. There is a `MirroredStrategy()` that can be used to train on a single machine with multiple GPUs but if you want to distribute training on multiple machines in a cluster/s(recommended and my goal), then `MultiWorkerMirroredStrategy()` strategy is a way to go.

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
```

The `num_replicas_in_sync` equals the number of devices that are used in the **all-reduce** operation. Use the `tf.distribute.MultiWorkerMirroredStrategy` API and with the help of this strategy, a keras model that was designed to run on a single worker can seamlessly work on multiple workers with minimal code changes.

#### What happens actually under the hood when this strategy is used?

1. Each GPU performs the forward pass on a different slice of the input data and computes the loss

2. Next each GPU compute the gradients based on the loss

3. These gradients are then aggregated across all of the devices(using an all-reduce algorithm)

4. The optimizer updates the weights using the reduced gradients thereby keeping the devices in sync

> [!NOTE]
> PyTorch has [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP](https://pytorch.org/docs/stable/fsdp.html)(more popular and useful)

I'm enabling automatic data sharding across workers by setting `tf.data.experimental.AutoShardPolicy` to `AutoShardPolicy.DATA`. This setting is needed to ensure convergence and performance. The concept of [sharding](https://www.Tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions) means handing each worker a subset of the entire dataset.

Now the final training workflow can be written below

```python
with strategy.scope():
    dataset = get_dataset().batch(BATCH_SIZE).repeat()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    model = build_and_compile_cnn_model()

model.fit(dataset, epochs=5, steps_per_epoch=70)
```

## Create a simple neural net

A simple neural net with `Adam` optimizer and `SparseCategoricalCrossentropy` loss as we have 10 categories to predict from.

```python
def build_and_compile_cnn_model():
    print("Training a simple neural net")

    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model
```

Let's define some necessary callbacks that will be executed during model training.

1. Checkpointing saves model weights at some frequency(use `save_freq`). We use `tf.keras.callbacks.ModelCheckpoint` for checkpointing.

```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
```

I define the checkpoint directory to store the checkpoints and the names of the files. Checkpoints are important to restore the weights if the model training stops due to some issues.

2. `tf.keras.callbacks.TensorBoard` writes a log for TensorBoard, which allows you to visualize the graphs.
   
3. `tf.keras.callbacks.LearningRateScheduler` schedules the learning rate to change after, for example, every epoch/batch.

```python
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5
```

4. PrintLR prints the learning rate at the end of each epoch.

```python
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(        epoch + 1, model.optimizer.lr.numpy()))
```

Now put all the components together.

```python
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]
```

Now every piece is in it's correct place.

Next, we train the model.

```python
model = build_and_compile_cnn_model()
model.fit(dataset, epochs=3, steps_per_epoch=70, callbacks=callbacks)
```

I'm getting an accuracy of 94% on the training data. I'm not spending much time on increasing the accuracy as it's not our end goal.

> [!NOTE]
> I'm doing these experiments in a colab notebook

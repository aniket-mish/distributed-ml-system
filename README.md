![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/7e8a65eb-c249-4ce7-9f4f-dcaaa6077fed)# Distributed Machine Learning on Kubernetes

Building a distributed system is the hottest skill that any swe/mle should have. Every tech company that uses ML(LLMs) wants to *serve* their customers at scale and if you're GPU rich and don't want to waste the resources by keeping the GPUs idle, you need to *train* the models in parallel. 

Recently, I was involved in a project where we developed a distributed scalable ML service which served hundreds of thousands of customers. TBH we faced a lot of challenges and building a distributed system is very tough. So i started working on this mini project to deepen my understanding of distributed ML and Kubernetes.

I want to actively develop and maintain this repo furthur, add monitoring and use aws for deployments.


![Image](https://cdn.kastatic.org/ka-perseus-images/0db827a36e9287ee9c130cf17610faaed276b931.svg)


## Introduction

### What are distributed systems?

Distributed systems are a group of nodes that talk to each other to achieve a specific task, such as streaming movies across devices, search engines, etc. 

### Why use distributed machine learning systems?

I wonder how these complex models with millions or billions of parameters are trained and served. The trick is to use distributed systems. They allow developers to handle massive datasets across multiple clusters, use automation tools, and benefit from hardware accelerations.

This repository includes code and references to implement a scalable and reliable machine learning system. I'm automating machine learning tasks with kubernetes, argo workflows, kubeflow, and tensorflow. I aim to construct machine learning pipelines that do data ingestion, distributed training, model serving, managing, and monitoring these workloads.

In this project i'm building an image classification end-to-end machine learning system.

The steps involved are:
1. Setup
2. Data ingestion
3. Distributed training
4. Prediction/evaluation
5. Serving
6. End-to-end workflow


## Setup

I'm using a macbook and homebrew to install the tools. If you're on linux/windows feel free to check the documentation of these tools for installation. We will install tensorflow, docker desktop, kubectl, and k3d.

[1] We will be using [tensorflow](https://www.tensorflow.org) for data processing, model building and evaluation.

```bash
pip install tensorflow
```

[2] We need docker to create single- or multi-node clusters. You can learn about docker [here](https://docker-curriculum.com/#setting-up-your-computer). I have used docker desktop here.

[3] Next, we install kubectl. It's a must have CLI tool for kubernetes.

```bash
brew install kubectl
```

[4] Next, to use kubernetes as our core distributed infrastructure, we will have to install [k3d](https://k3d.io/v5.5.2/) which is a lightweight wrapper to run k3s (Rancher Lab’s minimal kubernetes distribution) in docker. It's great for local kubernetes development. It's very lean and memory efficient. 

```bash
wget -q -O - https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=v5.0.0 bash
```

Next, we can create a cluster.

```bash
k3d cluster create dist-ml --image rancher/k3s:v1.25.3-k3s1
kubectl get nodes
```

![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/30b17dcb-2632-4812-93cb-31b783c52b11)


[5] We also install [kubectx](https://github.com/ahmetb/kubectx/) and kubens to easily switch contexts and namespaces.

```bash
brew install kubectx
```

[6] We will use kubeflow to submit jobs to the k8s cluster. To do this we install kubeflow training operator.

We start with creating a namespace. The namespaces provide a mechanism for isolating groups of resources within a single cluster. Read about the best practices to create and organize namespaces [here](https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-organizing-with-namespaces).

To create a namespace and name it _kubeflow_.

```bash
kubectl create namespace kubeflow
```

Next, switch the current context (default) to kubeflow

```bash
kubens kubeflow
```

> [!NOTE]
> I'm getting an error `couldn't get resource list for metrics.k8s.io/v1beta1: the server is currently unable to handle the request`. After looking on I understood that I need to edit the metrics server deployment yaml and add `hostNetwork: true` after `dnsPolicy`. It started working again.


<img width="668" alt="image" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/7103ff40-8f0e-42d3-b49e-0a2cd1b776e1">


Now, we install the dependencies for the Kubeflow training operator. This training operator provides Kubernetes custom resources that make running distributed or non-distributed TensorFlow jobs easy on Kubernetes.

```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.5.0"
```

<img width="1273" alt="image" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/022bca6c-c764-48a6-af4e-734a3465775f">


[7] We will install [argo workflows](https://argo-workflows.readthedocs.io/en/stable/) to construct end-to-end machine learning workflows.


![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/afbac898-117f-4800-917d-00cf848fc502)


## What are Pods?

This is the most atomic part of kubernetes ecosystem.

Let's create a simple kubernetes pod. You create a `hello-world.yaml` file as below.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: whalesay
spec:
  containers:
  - name: whalesay
    image: docker/whalesay:latest
    command: [cowsay]
    args: ["Hello world"]
```

Next, submit the job to our cluster.

```bash
kubectl create -f hello-world.yaml
```

We can see the status of the pod.

```bash
kubectl get pods
```

We can see what is being printed out in the container.

```bash
kubectl logs whalesay
```

If you want to get the details of a single pod with the raw YAML, then enter the following command.

```bash
kubectl get pod whalesay -o yaml
```

You can get the JSON or any other format as well.


## System Architecture

The system includes a distributed model training pipeline and an inference service that can be autoscaled. This will be automated using argo. There are different design patterns that one can choose from and which fits their needs.


<img width="1143" alt="Screenshot 2023-06-30 at 12 50 13 PM" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/18bb1322-1970-4ef4-a3a6-f7d345623ee0">


## Data Ingestion

We will use the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:


![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/9356978d-f3ab-4404-b35b-d5e50b3c82cb)


Here, 60,000 images are used to train the network and 10,000 images are used to evaluate how accurately the network learned to classify images.

### Single-node Data Pipeline

The `tf.data` API enables you to build complex input pipelines from simple, reusable pieces. It's very efficient. It enables handling large amounts of data, reading from different data formats, and performing complex transformations.

Load the fashion-mnist dataset into a `tf.data.Dataset` object and do some preprocessing(casting to float32). Next, we normalize the image pixel values from the [0, 255] to the [0, 1] range. We are keeping an *in-memory cache* to improve performance. We also shuffle the training data.

```python
import tensorflow_datasets as tfds
import tensorflow as tf

def mnist_dataset():
    BUFFER_SIZE = 10000
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train = datasets['train']
    return mnist_train.map(scale).cache().shuffle(BUFFER_SIZE)
```

We have used the tensorflow_datasets module which contains a collection of datasets ready to use. This gives us a shuffled dataset where each element consists of images and labels.

### Distributed Data Pipeline

We can consume our dataset in a distributed fashion as well and to do that we can use the same function we created before with some tweaks. When training a model with multiple GPUs, you can use the extra computing power effectively by increasing the batch size. In general, use the largest batch size that fits the GPU memory.

There is a `MirroredStrategy()` for use on a single machine with multiple GPUs. However, if you wish to distribute training on multiple machines in a cluster(our goal), then `MultiWorkerMirroredStrategy()` strategy is the way to go.

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
```

The `num_replicas_in_sync` equals the number of devices that are used in the [all-reduce]() operation. We have used the `tf.distribute.MultiWorkerMirroredStrategy` API and with the help of this strategy, a keras model that was designed to run on a single worker can seamlessly work on multiple workers with minimal code changes.

What actually happens behind the scenes?
1. Each GPU performs the forward pass on a different slice of the input data to compute the loss.
2. Each GPU computes the gradients based on the loss function.
3. These gradients are aggregated across all of the devices, via an all-reduce algorithm.
4. The optimizer updates the weights using the reduced gradients thereby keeping the devices in sync.

PyTorch users can look into the [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP](https://pytorch.org/docs/stable/fsdp.html) training approaches.

We have also enabled automatic data sharding across workers by setting `tf.data.experimental.AutoShardPolicy` to `AutoShardPolicy.DATA`. This setting is needed to ensure convergence and performance. Sharding means handing each worker a subset of the entire dataset. You can read more about it [here](https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions).

So the workflow is as below.

```python
with strategy.scope():
    dataset = mnist_dataset().batch(BATCH_SIZE).repeat()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    model = build_and_compile_cnn_model()

model.fit(dataset, epochs=3, steps_per_epoch=70)
```

## Model Training

### Single node Model Training

In the last step, we have created a distributed data ingestion component and have enabled data sharding as well.

```python
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
```

Next, we create a simple CNN model and instantiate the `Adam` optimizer. We are using accuracy to evaluate the model and sparse categorical cross entropy as the loss function(remember we have 10 categories to predict).

We also define callbacks(necessary ones 😉) that will execute during model training.

1. `tf.keras.callbacks.ModelCheckpoint` saves the model at a certain frequency, for example, after every epoch.

```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
```

We are defining the checkpoint directory to store the checkpoints and the name for the files. Checkpoints are important to restore the weights if the model training stops due to some issues.

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

We put together all the callbacks.

```python
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]
```

Next, we can train the model.

```python
single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(dataset, epochs=3, steps_per_epoch=70, callbacks=callbacks)
```

After training, we get an accuracy of 94%(meh) on the training data.


### Distributed Model Training

Next, we can insert the distributed training logic so that we can train the model on multiple workers. We use the MultiWorkerMirroredStrategy with keras(tf as backend).

In general, there are two common ways to do [distributed training with data parallelism](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras):
1. *Synchronous* training, where the steps of training are synced across the workers and replicas. Here, all workers train over different slices of input data in sync, and aggregating gradients at each step.
2. *Asynchronous* training, where the training steps are not strictly synced. Here, all workers are independently training over the input data and updating variables asynchronously. For instance, [parameter server training](https://www.tensorflow.org/tutorials/distribute/parameter_server_training).

We are using the MultiWorkerMirroredStrategy which implements synchronous distributed training across multiple workers, each with potentially multiple GPUs. It replicates all variables and computations to each local device and uses distributed collective implementation (e.g. all-reduce) so that multiple workers can work together.

Once we define our distributed training strategy, we initiate our distributed input data pipeline and the model inside the strategy scope.

## Model saving and loading

To save the model using `model.save`, the saving destinations(temporary dirs) need to be different for each worker.

- For non-chief(slave) workers, save the model to a temporary directory
- For the chief(master), save the model to the provided directory
The temporary directories of the workers need to be unique to prevent errors. The model saved in all the directories is identical, and only the model saved by the chief should be referenced for restoring or serving.

We will not save the model to temporary directories as doing this will waste our computing resources and memory. We will determine which worker is the chief and save its model only.

We can determine if the worker is the chief or not using the environment variable `TF_CONFIG`. Here's an example configuration:

```python
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
```
The `_is_chief` is a utility function that inspects the cluster spec and current task type and returns True if the worker is the chief and False otherwise.

```python
def _is_chief():
    return TASK_INDEX == 0

tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
TASK_INDEX = tf_config['task']['index']

if _is_chief():
    model_path = args.saved_model_dir
else:
    model_path = args.saved_model_dir + '/worker_tmp_' + str(TASK_INDEX)

multi_worker_model.save(model_path)
```

## Containerization 📦

We put everything we wrote till now into a python script called `multi-worker-distributed-training.py`. Next, we dockerize the app.

```dockerfile
FROM python:3.9
RUN pip install tensorflow==2.12.0 tensorflow_datasets==4.9.2
COPY multi-worker-distributed-training.py /
```

We then build the docker image.

```bash
docker build -f Dockerfile -t kubeflow/multi-worker-strategy:v0.1 .
```

<img width="865" alt="image" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/f4a6fbdb-0704-4f61-963d-b876874a2183">


Next, import the above image to the k3d cluster as it cannot access the image registry.

```bash
k3d image import kubeflow/multi-worker-strategy:v0.1 --cluster dist-ml
```


![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/9a6d843b-7cfd-451c-8c0a-ef08b8649326)


When we train our model in the k8s pods, once they are completed/failed, all files in the pods are recycled by the kubernetes garbage collecter. So all the model checkpoints are lost and we don't have a model for serving. To avoid this we need to use PersistentVolume(PV) and PersistentVolumeClaim(PVC).

A *_PersistentVolume (PV)_* is a piece of storage in the cluster that has been provisioned by an administrator or dynamically provisioned. It is a resource in the cluster just like a node is a cluster resource. PVs are volume plugins like Volumes but have a lifecycle independent of any individual Pod that uses the PV. This means that PV will persist and live even when the pods are deleted.

A *_PersistentVolumeClaim (PVC)_* is a request for storage by a user. It is similar to a Pod. Pods consume node resources and PVCs consume PV resources. Pods can request specific levels of resources (CPU and Memory). Claims can request specific size and access modes (e.g., they can be mounted ReadWriteOnce, ReadOnlyMany, or ReadWriteMany).

We can create a PVC to submit a request for storage that will be used in worker pods to store the trained model. Here we are requesting 1GB storage with ReadWriteOnce mode.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: strategy-volume
spec:
  accessModes: [ "ReadWriteOnce" ]
  volumeMode: Filesystem
  resources:
    requests:
      storage: 1Gi
```

Next, we create the PVC.

```bash
kubectl create -f multi-worker-pvc.yaml
```

Next, we will define a TFJob(model training w/tf) specification with the image we built before that contains the distributed training script.

```yaml
apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: multi-worker-training
spec:
  runPolicy:
    cleanPodPolicy: None
  tfReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: tensorflow
              image: kubeflow/multi-worker-strategy:v0.1
              imagePullPolicy: IfNotPresent
              command: ["python", "/multi-worker-distributed-training.py", "--saved_model_dir", "/trained_model/saved_model_versions/2/", "--checkpoint_dir", "/trained_model/checkpoint", "--model_type", "cnn"]
              volumeMounts:
                - mountPath: /trained_model
                  name: training
              resources:
                limits:
                  cpu: 500m
          volumes:
            - name: training
              persistentVolumeClaim:
                claimName: strategy-volume
```

We pass `saved_model_dir` and `checkpoint_dir` to the container. The `volumes` field specifies the persistent volume claim and `volumeMounts` field specifies what folder to mount the files. The `CleanPodPolicy` in the TFJob spec controls the deletion of pods when a job terminates. The `restartPolicy` determines whether pods will be restarted when they exit.

Next, we submit this TFJob to our cluster and start our distributed model training.

```bash
kubectl create -f multi-worker-tfjob.yaml
```


![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/aafe0c12-6b1e-4755-b49a-932c3c0214ca)


We can see 2 pods running our distributed training(we specified 2 workers).
1. multi-worker-training-worker-0
2. multi-worker-training-worker-1

We can see the logs from the pod `multi-worker-training-worker-0`.

```bash
kubectl logs multi-worker-training-worker-0
```


![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/24aabb20-68e9-4259-b892-11692c3f7dbb)


I have trained a CNN model and stored it in the `/saved_model_versions/1/` path. If we want, we can edit/update the code and resubmit the job. To do this you need to delete the running job, rebuild the docker image, import it, and resubmit the job.

```bash
kubectl delete tfjob --all; docker build -f Dockerfile -t kubeflow/multi-worker-strategy:v0.1 .; k3d image import kubeflow/multi-worker-strategy:v0.1 --cluster dist-ml; kubectl create -f multi-worker-tfjob.yaml
```

Next, evaluate the model's performance.

```bash
kubectl create -f predict-service.yaml
```

Now we have a trained model stored.

```bash
kubectl exec --stdin --tty predict-service -- bin/bash
```

We enter into a running container `predict-service`. It has the trained model stored at `trained_model/saved_model_versions/2/`.


![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/d61d86da-9188-4695-a166-9206e947a869)


Next, execute predict-service.py which takes the trained model and evaluates it on the test dataset.


## Model Selection

We've implemented the distributed model training component. In production, we might need to train different models and pick the top performer for model serving. Let's create two more models to understand this concept.

One model I'm trying is the CNN model with a batch norm layer.

```python
def build_and_compile_cnn_model_with_batch_norm():
    print("Training CNN model with batch normalization")
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1), name='image_bytes'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('sigmoid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('sigmoid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
```

The other model I'm trying is the CNN model with a dropout.

```python
def build_and_compile_cnn_model_with_dropout():
    print("Training CNN model with dropout")
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1), name='image_bytes'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
```

We train these models by submitting three different TFJobs with an argument `--model_type`.

To start training different models, update the `--model_type` and the `--saved_model_dir`, delete the currently running jobs and resubmit them.

```bash
kubectl delete tfjob --all
kubectl apply -f multi-worker-tfjob.yaml
```

Next, we load the testing data and the trained model to evaluate its performance. The model with the highest accuracy score can be moved to a different folder and used for model serving.

```python
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

best_model_path = ""
best_accuracy = 0

for i in range(3):
    model_path = "trained_models/saved_model_versions/" + str(i)
    model = tf.keras.models.load_model(model_path)

    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_test = datasets['test']
    ds = mnist_test.map(scale).cache().shuffle(BUFFER_SIZE).batch(64)
    loss, accuracy = model.evaluate(ds)

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_model_path = model_path

dst = "trained_model/saved_model_versions/3"
shutil.copytree(best_model_path, dst)
```

We add this script to the Dockerfile, rebuild the image, and create a pod that runs the script for model selection.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-selection
spec:
  containers:
  - name: predict
    image: kubeflow/multi-worker-strategy:v0.1
    command: ["python", "/model-selection.py"]
    volumeMounts:
    - name: model
      mountPath: /trained_model
  volumes:
  - name: model
    persistentVolumeClaim:
      claimName: strategy-volume
```


![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/4243d6c8-ce59-44c9-ad78-992f88aa7f5a)


## Model Serving

We implemented distributed training and model selection. Now we implement the model serving component. Here we take the trained model from `trained_model/saved_model_versions/3`. The model serving should be very performant.

### Single server model inference

```python
model_path = "trained_models/saved_model_versions/3"
model = tf.keras.models.load_model(model_path)
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_test = datasets['test']
ds = mnist_test.map(scale).cache().shuffle(BUFFER_SIZE).batch(64)
loss, accuracy = model.predict(ds)
```

We can also use [TFServing](https://keras.io/examples/keras_recipes/tf_serving/) to expose our model as an endpoint service. You can check the installation process [here](https://www.tensorflow.org/tfx/serving/setup).

```bash
# Environment variable with the path to the model
os.environ["MODEL_PATH"] = f"{model_path}"

nohup tensorflow_model_server \
  --port=8500 \
  --rest_api_port=8501 \
  --model_name=model \
  --model_base_path=$MODEL_PATH
```

_Nohup, short for no hang-up is a command in Linux systems that keeps processes running even after exiting the shell or terminal._

### Distributed model inference

The method mentioned above works great if we're only experimenting locally. There are more efficient ways for distributed model serving.

TensorFlow models contain a signature definition that defines the signature of a computation supported in a TensorFlow graph. SignatureDefs aims to provide generic support to identify the inputs and outputs of a function. We can modify this input layer with a preprocessing function so that clients can use base64 encoded images, which is a standard way of sending images through RESTFUL APIs. To do that, we’ll save a model with new serving signatures. The new signatures use Python functions to handle preprocessing the image from a JPEG to a Tensor. [Refer](https://cloud.google.com/blog/topics/developers-practitioners/add-preprocessing-functions-tensorflow-models-and-deploy-vertex-ai)

```python
def _preprocess(bytes_inputs):
    decoded = tf.io.decode_jpeg(bytes_inputs, channels=1)
    resized = tf.image.resize(decoded, size=(28, 28))
    return tf.cast(resized, dtype=tf.uint8)

def _get_serve_image_fn(model):
    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string, name='image_bytes')])
    def serve_image_fn(bytes_inputs):
        decoded_images = tf.map_fn(_preprocess, bytes_inputs, dtype=tf.uint8)
        return model(decoded_images)
    return serve_image_fn

signatures = {
    "serving_default": _get_serve_image_fn(model).get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='image_bytes')
    )
}

tf.saved_model.save(multi_worker_model, model_path, signatures=signatures)
```

Now we have updated the training script, we should rebuild the image and re-train the model.

Next, we will use KServe for inference service. [KServe](https://www.kubeflow.org/docs/external-add-ons/kserve/kserve/) enables serverless inferencing on Kubernetes and provides performant, high-abstraction interfaces for common machine learning (ML) frameworks like TensorFlow, PyTorch, etc. [Refer](https://kserve.github.io/website/0.11/modelserving/v1beta1/tensorflow/).

We create an [InferenceService](https://kserve.github.io/website/0.11/get_started/first_isvc/#run-your-first-inferenceservice) yaml, which specifies the framework tensorflow and storageUri that is pointed to a saved Tensorflow model.

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: InferenceService
metadata:
  name: tf-mnist
spec:
  predictor:
    model:
      modelFormat:
        name: tensorflow
      storageUri: "pvc://strategy-volume/saved_model_versions"
```

Install KServe.

```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.11/hack/quick_install.sh" | bash
```

Next, apply the inference-service.yaml to create the InferenceService. By default, it exposes an HTTP/REST endpoint.

```bash
kubectl apply -f inference-service.yaml
```

Wait for the InferenceService to be in a ready state.

```bash
kubectl get isvc tf-mnist
```

Next, we run the prediction. But first, we need to determine and set the INGRESS_HOST and INGRESS_PORT. An ingress gateway is like an API gateway that load-balances requests. To test it locally we have to do `Port Forward`.

```bash
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80
```

Then do the following in a different terminal window.

```bash
export INGRESS_HOST=localhost
export INGRESS_PORT=8080
```

We can send a sample request to our inference service. We can curl.

```bash
MODEL_NAME=tf-mnist
INPUT_PATH=@./mnist-input.json
SERVICE_HOSTNAME=$(kubectl get inferenceservice $MODEL_NAME -n kubeflow -o jsonpath='{.status.url}' | cut -d "/" -f 3)
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/$MODEL_NAME:predict -d $INPUT_PATH
```

or we use the requests library.

```python
input_path = "mnist-input.json"

with open(input_path) as json_file:
    data = json.load(json_file)

response = requests.post(
    url="http://localhost:8080/v1/models/tf-mnist:predict",
    data=json.dumps(data),
    headers={"Host": "tf-mnist.kubeflow.example.com"},
)
print(response.text)
```


![image](https://github.com/aniket-mish/distributed-ml-system/assets/71699313/6aa90d74-33f8-4f37-8b21-687e8c4453b0)


Our inference service is working as expected.


## Replicated model servers inference

Next, I want to have multiple model servers to handle large amounts of traffic. KServe can autoscale based on the requests. The autoscaler can scale down to zero if the application is receiving no traffic or we can specify a minimum number of replicas that need to be there. The `autoscaling.knative.dev/target` sets a soft limit. Other specs that can be configured like `minReplicas`, `containerConcurrency`, and `scaleMetric`, etc.

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: InferenceService
metadata:
  name: tf-mnist
  annotations:
    autoscaling.knative.dev/target: "1"
spec:
  predictor:
    model:
      modelFormat:
        name: tensorflow
      storageUri: "pvc://strategy-volume/saved_model_versions"
```

Next, I install [Hey](https://github.com/rakyll/hey), a tiny program that sends some load to a web application. Hey runs provided a number of requests in the provided concurrency level and prints stats.

```bash
# https://github.com/rakyll/hey
brew install hey
kubectl create -f inference-service.yaml

hey -z 30s -c 5 -m POST -host ${SERVICE_HOSTNAME} -D mnist-input.json "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/tf-mnist:predict"
```


<img width="1403" alt="Screenshot 2023-10-03 at 4 29 11 PM" src="https://github.com/aniket-mish/distributed-ml-system/assets/71699313/9d41eaac-4cf6-4649-8198-ca9daa2c3bb8">



I'm sending traffic for 30 seconds with 5 concurrent requests. As the scaling target is set to 1 and we load the service with 5 concurrent requests, the autoscaler tries scaling up to 5 pods. There will be a cold start time initially to spawn pods. It may take longer (to pull the docker image) if is not cached on the node.


## End-to-end Workflow

It's time to connect all the parts. I'm using argo workflow to orchestrate the jobs we executed before in an end-to-end fashion. We can build a CICD workflow using DAG (exactly similar to GitLab CICD) on Kubernetes. Argo is the defacto engine for orchestration on Kubernetes.

We will start by installing argo workflows in a different namespace.

```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.11/install.yaml
```

I'm creating an end-to-end workflow with 4 steps:
1. Data Ingestion
2. Distributed Training
3. Model Selection
4. Model Serving

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow                  # new type of k8s spec
metadata:
  generateName: tfjob-wf-    # name of the workflow spec
spec:
  entrypoint: tfjob-wf          # invoke the tfjob template
  templates:
  - name: tfjob-wf
    steps:
    - - name: data-ingestion-step
        template: data-ingestion-step
    - - name: distributed-tf-training-steps
        template: distributed-tf-training-steps
    - - name: model-selection-step
        template: model-selection-step
    - - name: create-model-serving-service
        template: create-model-serving-service
podGC:
  strategy: OnPodSuccess
volumes:
- name: model
  persistentVolumeClaim:
    claimName: strategy-volume
```

This is a multi-step workflow where all the steps are executed sequentially(double dash). `PodGC` describes how to delete completed pods. Deleting completed pods can free the resources. I'm also using persistent storage to store the dataset and the trained models.

The first step is the data ingestion. We have added a `memoize` spec to cache the output of this step. Memoization reduces cost and execution time. Since we do not want to download the data every time, we can cache it using the configMap. We have to specify the `key` and name for the `config-map` cache. I have also specified `maxAge` to `1h`, which defines how long should the cache be considered valid.

```yaml
- name: data-ingestion-step
  serviceAccountName: argo
  memoize:
  cache:
    configMap:
      name: data-ingestion-config
      key: "data-ingestion-cache"
    maxAge: "1h"
  container:
    image: kubeflow/multi-worker-strategy:v0.1
    imagePullPolicy: IfNotPresent
    command: ["python", "/data-ingestion.py"]
```

Next, we execute the model training steps in parallel.


```yaml
- name: distributed-training-step
  steps:
  - - name: cnn-model
      template: cnn-model
    - name: cnn-model-with-dropout
      template: cnn-model-with-dropout
    - name: cnn-model-with-batch-norm
      template: cnn-model-with-batch-norm
```

Next, we create a step to run distributed training with the CNN model. To create the TFJob, we include the manifest we created before. We also add the `successCondition` and `failureCondition` to indicate if the job is created. Here we are storing the trained model in a different folder. We create similar steps for the other two models.


```yaml
- name: cnn-model
  serviceAccountName: training-operator
  resource:
    action: create
    setOwnerReference: true
    successCondition: status.replicaStatuses.Worker.succeeded = 2
    failureCondition: status.replicaStatuses.Worker.failed > 0
  manifests: |
    apiVersion: kubeflow.org/v1
    kind: TFJob
    metadata:
      generateName: multi-worker-training-
    spec:
      runPolicy:
        cleanPodPolicy: None
      tfReplicaSpecs:
        Worker:
          replicas: 2
          restartPolicy: Never
          template:
            spec:
              containers:
                - name: tensorflow
                  image: kubeflow/multi-worker-strategy:v0.1
                  imagePullPolicy: IfNotPresent
                  command: ["python", "/multi-worker-distributed-training.py", "--saved_model_dir", "/trained_model/saved_model_versions/1/", "--checkpoint_dir", "/trained_model/checkpoint", "--model_type", "cnn"]
                  volumeMounts:
                    - mountPath: /trained_model
                      name: training
                  resources:
                    limits:
                      cpu: 500m
              volumes:
                - name: training
                  persistentVolumeClaim:
                    claimName: strategy-volume
```

Next, we add the model selection step. It is similar to `model-selection.yaml` we created earlier.

```yaml
- name: model-selection-step
  serviceAccountName: argo
  container:
    image: kubeflow/multi-worker-strategy:v0.1
    imagePullPolicy: IfNotPresent
    command: ["python", "/model-selection.py"]
    volumeMounts:
    - name: model
      mountPath: /trained_model
```

The last step of the workflow is the model serving.

```yaml
- name: create-model-serving-service
  serviceAccountName: training-operator
  successCondition: status.modelStatus.states.transitionStatus = UpToDate
  resource:
    action: create
    setOwnerReference: true
    manifest: |
      apiVersion: "serving.kserve.io/v1beta1"
      kind: InferenceService
      metadata:
        name: tf-mnist
      spec:
        predictor:
          model:
            modelFormat:
              name: tensorflow
            storageUri: "pvc://strategy-volume/saved_model_versions"
```

Next, run the workflow.

```bash
kubectl create -f workflow.yaml
```

## Logger

Logging is an essential component of the machine learning system. It helps debug issues, analyze performance, troubleshoot errors, gather insights, and implement a feedback loop. Fortunately, KServe makes it easy to create a service called message-dumper. It logs the request and the response. It has a unique identifier for the request and the response.

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: message-dumper
spec:
  template:
    spec:
      containers:
      - image: gcr.io/knative-releases/knative.dev/eventing-contrib/cmd/event_display
```

```bash
kubectl create -f message-dumper.yaml
```

Next, we include the logger which points to the message dumper URL in the InferenceService predictor.

```yaml
logger:
  mode: all
  url: http://message-dumper.default/
```

You can read about the inference logger [here](https://kserve.github.io/website/0.8/modelserving/logger/logger/#create-an-inferenceservice-with-logger).

## Monitoring

I am setting up Prometheus and Grafana to monitor the Kubernetes cluster's resources. 

Prometheus stores data in a time series fashion and Grafana provides dashboards for the visualization.

I am using helm charts here.

```bash
brew install helm
```

## Summary

1. A distributed machine learning system is designed to train machine learning models on large datasets that cannot be processed on a single machine. There is a need to distribute the computation or training process to train complex models with millions or rather billions of parameters.
2. Kubernetes is a popular choice for building such complex distributed systems. We can build scalable and highly available systems using K8s.
3. Tensorflow provides several strategies for distributed training. We have used `MultiWorkerMirroredStrategy` here.
4. We have used KServe for building an Inference Service which can be autoscaled based on the traffic.
5. Argo workflows help build CICD pipelines on Kubernetes.

## todos

- Monitoring the metrics with Prometheus and Grafana See [Monitoring](#Monitoring)
- Setup MLFlow tracking server See [MLFlow Tracking Server](#MLFlow)
- Try Seldon MLServer for serving [Seldon MLServer](#MLServer)
- Use Helm charts?
- Manage GPU workloads
- Create a PyTorch version
- Use terraform to create K8S cluster on EKS (cloud deployments?)

## References

[1] [Distributed Machine Learning Patterns by Yuan Tang](https://www.manning.com/books/distributed-machine-learning-patterns?utm_source=terrytangyuan&utm_medium=affiliate&utm_campaign=book_tang_distributed_6_10_21&a_aid=terrytangyuan&a_bid=9b134929)

[2] [Multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)

[3] [Distributed training with Keras](https://www.tensorflow.org/tutorials/distribute/keras)

[4] [Custom training loop with Keras and MultiWorkerMirroredStrategy](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl)

[5] [Distributed training with TensorFlow](https://www.tensorflow.org/guide/distributed_training)

[6] [First InferenceService](https://kserve.github.io/website/0.7/get_started/first_isvc/#5-run-performance-test)

[7] [Autoscale InferenceService with inference workload](https://kserve.github.io/website/0.8/modelserving/autoscaling/autoscaling)

[8] [Kubectl cheatsheet](https://www.bluematador.com/learn/kubectl-cheatsheet)

[9] [Load testing with Hey](https://github.com/rakyll/hey)

[10] [Argo Workflows](https://argoproj.github.io/argo-workflows/)

[11] [TensorFlow Distributed Training on Kubeflow](https://dzlab.github.io/ml/2020/07/18/kubeflow-training/)

[12] [MLOps for all](https://mlops-for-all.github.io/en/docs/introduction/intro)

[13] [How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/)

[14] [Getting Started with Argo Workflows](https://velocity.tech/blog/getting-started-with-argo-workflows)

[15] [Kubernetes best practices](https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-how-and-why-to-build-small-container-images)

[16] [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)

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


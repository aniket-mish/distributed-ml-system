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

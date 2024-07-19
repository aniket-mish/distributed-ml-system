FROM python:3.10.12-slim-buster

RUN pip install torch torchvision

COPY data-ingestion.py /
COPY distributed-training.py /

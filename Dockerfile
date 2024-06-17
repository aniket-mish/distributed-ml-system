FROM python:3.9

RUN pip install tensorflow tensorflow_datasets

COPY data-ingestion.py /
COPY distributed-training.py /

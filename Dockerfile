FROM python:3.10

RUN apt-get update && apt-get install -y libhdf5-dev

RUN pip install tensorflow tensorflow-datasets

COPY data_ingestion.py /
COPY distributed-training.py /
COPY model-selection.py /

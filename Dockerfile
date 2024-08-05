FROM python:3.9

# Install necessary dependencies
RUN apt-get update && apt-get install -y libhdf5-dev

RUN pip install tensorflow==2.12.0 tensorflow_datasets==4.9.2

COPY data_ingestion.py /
COPY predict-service.py /
COPY model-selection.py /
COPY distributed-training.py /
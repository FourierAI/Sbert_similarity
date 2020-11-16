FROM anibali/pytorch:1.5.0-cuda10.2

WORKDIR /app

COPY sentence_transformers /app/sentence_transformers
COPY /datasets /app/datasets
COPY train.py /app
COPY eval.py /app
COPY requirements.txt /app

RUN pip install -r requirements.txt

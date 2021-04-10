FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
LABEL maintainer="Jihwan Lim"

COPY requirements.txt /

RUN apt-get update -y && apt-get upgrade -y \
    && pip install --upgrade pip setuptools wheel \
    && apt-get install -y libhdf5-dev \
    && pip install -r /requirements.txt

COPY . /app
WORKDIR /app

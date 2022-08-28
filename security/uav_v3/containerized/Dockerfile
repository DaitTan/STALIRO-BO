# Specify BROKER_URL and QUEUE when running
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y curl ca-certificates amqp-tools python3 python3-pip \
       --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* 
RUN python3 -m pip install setuptools pip install torch==1.10.2+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
COPY . /
RUN python3 -m pip install -r requirements.txt

CMD  /usr/bin/amqp-consume --url=$BROKER_URL -q $QUEUE -c 1 /worker.py
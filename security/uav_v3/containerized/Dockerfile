# Specify BROKER_URL and QUEUE when running
FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y curl ca-certificates amqp-tools python3 python3-pip \
       --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* 
COPY . /
RUN python3 -m pip install setuptools
RUN python3 -m pip install -r requirements.txt

CMD  /usr/bin/amqp-consume --url=$BROKER_URL -q $QUEUE -c 1 /worker.py
# Specify BROKER_URL and QUEUE when running
FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y curl ca-certificates amqp-tools python3 \
       --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
COPY . /

CMD  /usr/bin/amqp-consume --url=$BROKER_URL -q $QUEUE -c 1 /worker.py
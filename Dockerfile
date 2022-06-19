FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip python3-pyqt5
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
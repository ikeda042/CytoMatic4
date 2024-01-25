FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt .

RUN pip3 install -r requirements.txt

COPY . /app

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends apache2-utils
RUN apt -y update && apt -y upgrade
RUN apt -y install libopencv-dev
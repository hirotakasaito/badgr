FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

COPY requirements.txt .
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python3 python3-pip

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /root

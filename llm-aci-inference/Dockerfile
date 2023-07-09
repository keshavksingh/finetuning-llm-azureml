#https://hub.docker.com/r/huggingface/transformers-pytorch-gpu/tags
FROM huggingface/transformers-pytorch-gpu:latest

FROM python:3.8.2

WORKDIR /

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["/bin/bash","./startup.sh"]
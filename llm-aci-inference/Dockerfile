FROM ubuntu:latest

USER root

# Install Anaconda
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y wget && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

#FROM continuumio/miniconda3
RUN conda update conda
COPY requirements.yaml /tmp/requirements.yaml
RUN conda env create --name llm-inference -f /tmp/requirements.yaml
RUN echo "source activate llm-inference" > ~/.bashrc
ENV PATH /opt/conda/envs/llm-inference/bin:$PATH
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

WORKDIR /
COPY . .

ENTRYPOINT ["/bin/bash", "./startup.sh"]

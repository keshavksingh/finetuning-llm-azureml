# Use nvidia/cuda image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY ./requirements.yaml /tmp/requirements.yaml
RUN conda update conda \
    && conda env create --name llm-inference -f /tmp/requirements.yaml

RUN echo "conda activate llm-inference" >> ~/.bashrc
ENV PATH /opt/conda/envs/llm-inference/bin:$PATH
ENV CONDA_DEFAULT_ENV $llm-inference

WORKDIR /
COPY . .

ENTRYPOINT ["/bin/bash","./startup.sh"]
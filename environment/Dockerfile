# hash:sha256:3c9d29a426b98761f4850f19761a51bc617b8ff57091ae3fa743e582b5f48642
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.2.1 \
    hdmf==3.14.6 \
    hdmf-zarr==0.11.3 \
    matplotlib==3.10.1 \
    numpy==2.2.5 \
    pandas==2.2.3 \
    pynwb==3.0.0

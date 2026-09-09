#Author : Paarth Sharma 
#File Name : Dockerfile
#Project Name : swiftJEPA
#Creation Date : 8th September 2026
#Modification Date : 8th September 2026
#Description : Single containerized development environment for swiftJEPA. It has a CUDA 12.6 base and torch cu126. 

#Base Image (CUDA 12.6 devel) shipped with nvcc which is required for kernel development later 
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Python and build headers 
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git && rm -rf /var/lib/apt/lists/*

# venv setup (Ubuntu 24.04 system python is PEP-668 which is externally-managed)
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch from the cu126 index to match nvcc toolkit
COPY requirements-cuda.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-cuda.txt 

# Python dependencies for the rest of the project 
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# directory for nvcc for the kernel build later 
ENV CUDA_HOME=/usr/local/cuda

# Matches compose binding so both have the same targets 
WORKDIR /workspace

# Set the default command line program to bash
CMD ["bash"]
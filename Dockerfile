#------------------------------------------------------------#
# Docker file for an MDI engine
#------------------------------------------------------------#

FROM taylorabarnes/devenv-cuda:latest

ENV DEBIAN_FRONTEND=noninteractive

# run the build script
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y \
                       vim \
                       python3 \
                       python3-pip \
                       python3-dev \
                       && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyCUDA and other Python packages
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
                          pycuda \
                          numpy \
                          scipy \
                          matplotlib \
                          pandas \
                          jupyter

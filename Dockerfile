#------------------------------------------------------------#
# Docker file for an MDI engine
#------------------------------------------------------------#

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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

# Copy the entrypoint file into the Docker image
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Define the entrypoint script that should be run
ENTRYPOINT ["/entrypoint.sh"]

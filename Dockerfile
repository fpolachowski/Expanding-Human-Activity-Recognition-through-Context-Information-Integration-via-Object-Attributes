ARG BASE_IMAGE=ubuntu:22.04

# Use Ubuntu base image
FROM ${BASE_IMAGE} as dev-base

# Install essential packages, including Python
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    libjpeg-dev \
    nano \
    libpng-dev \
    python3.10 \
    python3.10-dev \
    python3-pip && \
    apt install -y python3.10-distutils && \
    ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Set Python and pip environment variables
ENV PYTHONPATH="/usr/local/lib/python3.10/site-packages:$PYTHONPATH"

# Install Python packages directly using pip
COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip setuptools && \
    python3.10 -m pip install astunparse expecttest hypothesis numpy psutil pyyaml requests setuptools \
        types-dataclasses typing-extensions sympy filelock networkx jinja2 fsspec protobuf && \
    python3.10 -m pip install -r requirements.txt && \
    python3.10 -m pip install torch torchvision torchaudio && \
    rm -rf ~/.cache/pip

# Set up environment variables for WandB (or other configurations as needed)
ENV WANDB_API_KEY key
ENV WANDB_BASE_URL "url"

# Set working directory and configure Git settings
WORKDIR /workspace
RUN git config --global --add safe.directory /workspace

# Set the default command to start the training script
CMD ["python3.10", "src/train.py"]

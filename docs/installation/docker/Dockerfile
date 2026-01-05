ARG PYTORCH="2.7.1"
ARG CUDA="11.8"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0+PTX;7.5;8.0;8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV DEBIAN_FRONTEND=noninteractive

# Update and install OS packages.
RUN apt-get update && apt-get install -y wget vim git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root appuser.
# Obtain your USER_ID from the bash command `id -u`.
ARG USER_ID
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"
WORKDIR /home/appuser

# Install dependencies and useful packages.
RUN pip install --no-cache-dir --user --upgrade pip wheel
RUN conda clean -y --all

# Add the project and install.
COPY --chown=appuser:appuser . node-classification/
WORKDIR /home/appuser/node-classification
RUN pip install --no-cache-dir --user -e .

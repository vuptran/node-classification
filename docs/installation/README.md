# Installation Guide
We first clone the repository to our local home directory:

```bash
git clone \
    https://github.com/vuptran/node-classification.git \
    ~/node-classification
# We create some new directories inside the repo
# to store development artifacts.
cd ~/node-classification
mkdir results work_dirs
```

## Best Practices with Docker
We recommend using Docker to containerize all the complex dependencies when building this project. We provide a reference [Dockerfile](https://github.com/vuptran/node-classification/tree/main/docs/installation/docker/Dockerfile) as an example.

Make sure the GPU driver satisfies the minimum version requirements, according to [these NVIDIA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Also, ensure that the [docker version](https://docs.docker.com/engine/install/) is >=19.03.

### Build

```bash
# The reference build uses Python 3.11.13, PyTorch 2.7.1,
# PyTorch Geometric 2.6.1, PyTorch Lightning 2.5.4, and CUDA 11.8.
docker build \
    --build-arg USER_ID=$(id -u) \
    -t node-classification:pytorch2.7.1-cuda11.8-pyg2.6.1-lightning2.5.4 \
    -f docs/installation/docker/Dockerfile .
```

### Usage
We recommend running Docker as a user mapped from our local machine to the container via the argument `-u $(id -u)`, where the `bash` command `id -u` gives the user ID on the local host. Below is an example `docker run` command to execute an example training job.

```bash
LOCAL_HOME_DIR=~
APP_HOME_DIR=/home/appuser
DATA_DIR=/data
docker run \
    -w ${APP_HOME_DIR}/node-classification \
    --gpus='"device=0"' \
    -u $(id -u) --rm --ipc=host \
    -v ${LOCAL_HOME_DIR}/node-classification:${APP_HOME_DIR}/node-classification \
    -v ${DATA_DIR}:${APP_HOME_DIR}/node-classification/data \
    node-classification:pytorch2.7.1-cuda11.8-pyg2.6.1-lightning2.5.4 \
    python nodecls/train.py \
    --data-path ./data/ogbn/arxiv/ogbn-arxiv.npz --model-type gnn \
    --max-epochs 2000 --train-batch-size 169343 --test-batch-size 169343 \
    --num-classes 40 --num-workers 8 --eval-metric accuracy \
    --checkpoint-path ./work_dirs/ogbn_arxiv/gnn/ --hidden-channels 256 \
    --num-layers 4 --input-dropout 0.0 --dropout 0.5 --learning-rate 0.0005 \
    --num-devices 1 --test-after-train --test-on-gpu
```

Here, we assume that the data source is stored on the local machine at the path `/data/`. We use the `docker run -v` flag to map volumes between the local host and the container at runtime. Our recommended best practice is to map two volumes from the local host to be used by the container:

1. We map the entire local repository to the container so that any local modifications will take effect inside the container and can be used by the container at runtime.

    ```bash
    -v ${LOCAL_HOME_DIR}/node-classification:${APP_HOME_DIR}/node-classification
    ```

2. We map *data volumes* to a specified location inside the container so it can access data not previously copied during runtime.

    ```bash
    -v ${DATA_DIR}:${APP_HOME_DIR}/node-classification/data
    ```

Then we just need to point to the proper paths of the data source and other artifacts needed by the job, which are *relative to the working directory of the container*. The default working directory of the container is set by `docker run -w ${APP_HOME_DIR}/node-classification`.

## Anaconda Environment
Alternatively, we can install the project and its dependencies using an Anaconda environment.

**Step 1.** Download and install Anaconda from the [official website](https://www.anaconda.com/download).

**Step 2.** Create the conda environment.

```bash
# From inside the `node-classification` working directory.
conda env create -f docs/installation/environment.yaml
```

**Step 3.** Verify the installation.

```bash
conda activate node-classification

# No import error.
python nodecls/train.py -h
```
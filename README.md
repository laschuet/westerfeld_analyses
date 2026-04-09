# Westerfeld analyses

Accompanying repository. This repository contains the analysis scripts.

## Getting started

The following few steps are necessary to get started.

### Get the repository

Clone the repository with [Git](https://www.git-scm.com):
```
git clone https://github.com/laschuet/westerfeld_analyses.git
```

Alternatively,
[download](https://github.com/laschuet/westerfeld_analyses/archive/refs/heads/main.zip)
and unzip the repository as a file.

### Install Python dependencies

The scripts provided are written in [Python 3](https://www.python.org).

We recommend using [uv](https://docs.astral.sh/uv) for managing all
dependencies. Please follow the official [installation
instructions](https://docs.astral.sh/uv/getting-started/installation).

Create the project environment in the root directory of this repository:
```
uv sync
```

### Get the dataset

Get the [dataset](https://doi.org/10.20387/bonares-w669-gdsd) from the [BonaRes
Repository for Soil and Agricultural Research
Data](https://www.bonares.de/research-data).

Move the dataset into the root directory of this repository, i.e., the dataset
should be located next to this README.

## Analyses

Run the analyses:
```
uv run westerfeld/ncm.py
```

## Using Docker

We provide a pre-build docker image which contains all necessary dependencies. This requires a [docker installation](https://docs.docker.com/engine/install/).

Run the scripts usind docker run: 

**Analyses:**
```bash
docker run \
-v /path/to/data:/app/input \
-v /path/to/results:/app/out \
ghcr.io/laschuet/westerfeld_analyses:latest westerfeld/ncm.py
```

**Co-occurence network analysis:**
```bash
docker run \
-v /path/to/data:/app/input \
-v /path/to/results:/app/out \
ghcr.io/laschuet/westerfeld_analyses:latest westerfeld/cooccurence.py
```

> This maps both the data folder to `/app/input` and the results at `/app/out`.


### Building Docker Image Manually

The docker image can also be build locally:

```bash
docker build -t westerfeld-analyses -f docker/Dockerfile .
```

> To run this image, run the commands above but replace `ghcr.io/laschuet/westerfeld_analyses:latest` with `westerfeld-analyses`.

## License

This work is licensed under the [MIT License](./LICENSE.txt).

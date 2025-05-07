# AI Robustness Benchmark for Cyber-Physical Systems

*This repository is still a work in progress*

## Folder Structure
```
├── README.md          <- The top-level README for developers using this project.
│
├── data               <- Datasets and lightning datamodules.
│   └── processed      <- The final, canonical datasets for modeling.
│
├── (logdir)           <- Optional: Model checkpoints and results, if run locally.
│   
├── models             <- Lightning modules of all the models.
│
├── notebooks          <- Jupyter notebooks describing the datasets.
│
└── visualizations     <- Generated graphics and figures used in the paper.
```

## Getting Started

### Install the Dependencies

We use [uv](https://docs.astral.sh/uv/) as our package manager. If not already installed on your system, please install it first.

Once uv is installed on your system, install this project by running:
```bash
uv sync
```
at the top of this directory.

You can install an ipykernel which lets you use this environment inside Jupyter notebooks like so:
```bash
source .venv/bin/activate
python -m ipykernel install --user --name robust-env 
```

### Run the Experiments

#### Get the Data
The framework works on arbitrary time series datasets in CSV or parquet format.
To test the framework, we added a synthetic dataset describing a three tank system under './data/processed'.
The dataset has been proposed in [(Steude et al., 2022)](https://www.sciencedirect.com/science/article/pii/S2405896322004840) and was later refined in [(Windmann et al., 2023)](https://arxiv.org/abs/2306.07737) into the current version.

#### Use the Package
There are three main scripts that can be used to interact with our package:

1. `./run_training.py`
2. `./run_testing.py`
3. `./run_cleaning.py`

All of them do what their names promise (the cleaning script is optional and removes model
checkpoints from MLflow that are not needed anymore).

The training and testing scripts both accept command line arguments (click).
However, if not executed with any arguments, they will use the defaults
specified in the config file `./config.py`.

This includes the URL of the MLflow server.
We use MLflow for tracking the experiments.
If the MLflow server is configured to use an S3 compatible object store in the
backend, make sure to have the corresponding env variables in your environment,
specifically `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. These are also needed
if you want to read data from a remote S3 location.

So, to run the training on your local machine, you could do:
```bash
source .venv/bin/activate # if the env is not activated already
python run_training.py --model Informer
```

Similarly, to run the testing script, run:
```bash
source .venv/bin/activate # if the env is not activated already
python run_testing.py
```

## Parallelization using Kubeflow Pipelines
The Python package can be used as is on the infrastructure of your choice.
However, we decided to use [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) for scaling the hyperparameter search on
Kubernetes. This allows us to efficiently distribute and parallelize our experiments.

To use Kubeflow, we need container images. These can be built as follows:

### Build the Image:
First, export the requirements to a txt file (since we won't be using uv in the
container build):
```sh
uv export > requirements.txt
```
Then build the image like so (assuming you're using Docker):
```sh
docker build . --platform=linux/amd64 -t <your-registry>/<your-image-name>:<your-tag>
```

### Push the Image
For the image to be used in the cluster, it must be pushed to a remote registry. First, login to the registry:
```sh
docker login 
```

and then push the image:
```sh
docker push <your-registry>/<your-image-name>:<your-tag>
```

### Run the Pipeline
To run the pipeline on your cluster, replace the TRAINING_IMAGE string in the
`kf_pipeline_training.py` file and run it like so:
```bash
source .venv/bin/activate # if the env is not activated already
python ./kf_pipeline_training.py 
```

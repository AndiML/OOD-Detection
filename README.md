
# OOD-Detection Framework

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python Version](https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg)](https://www.python.org/downloads/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
![Contributors](https://img.shields.io/github/contributors/AndiML/Action-Recognition-System.svg)

---

## Overview

The **OOD-Detection** framework automates the process of Out-Of-Distribution (OOD) detection by providing a complete end-to-end pipeline that:

- **Downloads and configures datasets** for both in-distribution training and OOD evaluation.
- **Trains models** on specified in-distribution data.
- **Evaluates the trained models** on OOD samples using both internal and external partitioning strategies.
- **Logs experiments** comprehensively—including hyperparameters, training metrics, and OOD evaluation metrics—via an integrated experiment tracker.

This modular framework is designed for reproducibility and extensibility, making it easy to test different model architectures and hyperparameter settings.

---

## Table of Contents

- [OOD-Detection Framework](#ood-detection-framework)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Supported Models](#supported-models)
  - [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
    - [Example Commands](#example-commands)
      - [Training a Variational Autoencoder (VAE)](#training-a-variational-autoencoder-vae)
      - [Training a VAE with Noise](#training-a-vae-with-noise)
  - [Pipeline Details](#pipeline-details)
  - [Experiment Logging](#experiment-logging)
  - [Contributing](#contributing)
  - [License](#license)

---

## Installation

Clone the repository and create the conda environment using the provided `environment.yaml` file:

```bash
git clone git@github.com:AndiML/OOD-Detection.git
cd OOD-Detection
conda env create -f environment.yaml
```

If you add new packages later, update the environment file with:

```bash
conda env export | grep -v "prefix" | grep -v "numpy" > environment.yaml
```

---

## Supported Models

The framework is designed to support multiple neural network architectures for OOD detection. Currently, the supported models (defined in the module `ood_detection.src.models`) include, for example:

- **VAE (Variational Autoencoder):** A reconstruction-based model ideal for unsupervised learning.
- **VAE with Noise:** A variant of the VAE model that includes noise injection (controlled by the `--noise_std` parameter) for robustness testing.
- *(Additional models can be added by extending the `MODEL_IDS` and related model generator utilities.)*

---

## Usage

Before running experiments, activate the conda environment:

```bash
conda activate ood-detection
```

The main training and evaluation command is exposed via the `train-model` command. This command takes care of:

- Downloading and preparing datasets.
- Configuring the device (CPU or GPU).
- Creating the model based on the selected architecture.
- Setting up optimizers, schedulers, and training loops.
- Logging hyperparameters and metrics (including OOD scores) using the integrated `ExperimentLogger`.

### Command-Line Arguments

Key command-line arguments include:

- **Dataset and Partitioning:**
  - `output_path` – Directory for experiment results.
  - `dataset_path` – Directory where datasets are stored or downloaded.
  - `--in_dataset` – Name of the in-distribution dataset.
  - `--ood_datasets` – List of OOD dataset IDs.
  - `--partition_method` – Partitioning method: `"internal"` (reserve in-distribution classes) or `"external"` (use separate OOD datasets).
  - `--num_inliers` – For internal partitioning: number of inlier classes to reserve.

- **Training Settings:**
  - `--epochs` – Number of training epochs.
  - `--batchsize` – Batch size for training.
  - `--learning_rate`, `--momentum`, `--weight_decay` – Optimizer parameters.

- **Model and Experiment Settings:**
  - `--model_type` – Model architecture (e.g., `vae`).
  - `--latent_dim` – Dimensionality of the latent representation (for reconstruction-based models).
  - `--noise_std` – Noise standard deviation for experiments comparing a plain VAE versus a noisy VAE.
  - `--use_gpu` – Flag to enable CUDA.
  - Additional arguments for optimizer and scheduler configuration (e.g., `--optimizer`, `--scheduler`, `--step_size`, etc.).

### Example Commands

#### Training a Variational Autoencoder (VAE)

To run a standard VAE experiment on the MNIST dataset:

```bash
python -m ood_detection train-model \
  /path/to/output \
  /path/to/dataset \
  --in_dataset mnist \
  --model_type vae \
  --epochs 10 \
  --batchsize 64 \
  --learning_rate 0.001 \
  --latent_dim 100 \
  --use_gpu
```

This command will:
- Download/configure the MNIST dataset.
- Instantiate a VAE with a 100-dimensional latent space.
- Train the model for 10 epochs.
- Log training and OOD evaluation metrics.

#### Training a VAE with Noise

To compare performance with noise injection, run:

```bash
python -m ood_detection train-model \
  /path/to/output \
  /path/to/dataset \
  --in_dataset mnist \
  --model_type vae \
  --epochs 10 \
  --batchsize 64 \
  --learning_rate 0.001 \
  --latent_dim 100 \
  --noise_std 0.1 \
  --use_gpu
```

The additional `--noise_std 0.1` flag configures the model to add noise to the input images during training, which can be useful for assessing robustness.

---

## Pipeline Details

The core training logic is implemented in the `TrainModelCommand` and its descriptor:

- **Dataset Preparation:**
  Downloads and prepares the in-distribution dataset; creates training and validation loaders.

- **Model and Device Configuration:**
  Sets up the model (e.g., VAE) and configures the training device based on the `--use_gpu` flag.

- **Optimizer and Scheduler:**
  Configures optimizer (SGD, Adam, etc.) and learning rate schedulers based on command-line parameters.

- **Training and OOD Evaluation:**
  After model training, the pipeline partitions data either internally or externally to compute anomaly scores. These scores are then used to compute and log OOD metrics.

- **Experiment Logging:**
  The integrated `ExperimentLogger` saves hyperparameters, logs per-epoch metrics to CSV (and TensorBoard if enabled), and logs OOD evaluation details into partition-specific CSV files.

---

## Experiment Logging

The `ExperimentLogger` class provides detailed logging for every experiment:

- **Hyperparameter Logging:**
  Saves all key parameters (including the experiment start timestamp) to a YAML file.

- **Metric Tracking:**
  Logs training metrics (loss, accuracy) and OOD metrics (computed anomaly scores) to CSV files. Optionally, metrics are also sent to TensorBoard for visualization.

- **Image Logging:**
  For reconstruction-based models, both original and reconstructed images are logged either via TensorBoard or saved as files for later inspection.

This logging mechanism ensures that every experiment is fully reproducible and that all details are stored for future reference.

---

## Contributing

Contributions for additional commands, new model architectures, or improvements to experiment logging are welcome. Please ensure that
any new features align with the framework’s focus on reproducibility and comprehensive experiment tracking.

---

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

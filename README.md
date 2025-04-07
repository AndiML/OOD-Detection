
# OOD-Detection Framework

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python Version](https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg)](https://www.python.org/downloads/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
![Contributors](https://img.shields.io/github/contributors/AndiML/Action-Recognition-System.svg)

---

## Overview

The **OOD-Detection** framework automates the process of Out-Of-Distribution (OOD) detection for medical datasets by providing a complete end-to-end pipeline that:

- **Downloads and configures datasets** for both in-distribution training and OOD evaluation.
- **Trains models** on specified in-distribution data.
- **Evaluates the trained models** on OOD samples using both internal and external partitioning strategies.
- **Logs experiments** comprehensively—including hyperparameters, training metrics, and OOD evaluation metrics—via an integrated experiment logger.


## Table of Contents

- [OOD-Detection Framework](#ood-detection-framework)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Supported Models](#supported-models)
  - [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
    - [Example Commands](#example-commands)
      - [Training a Variational Autoencoder (VAE) for a Medical Imaging Task](#training-a-variational-autoencoder-vae-for-a-medical-imaging-task)
      - [Training with Internal Partitioning](#training-with-internal-partitioning)
      - [OOD Detection on Chest X-rays](#ood-detection-on-chest-x-rays)
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

The framework supports multiple neural network architectures for OOD detection. Currently, the supported models (defined in the module `ood_detection.src.models`) include, for example:

- **VAE (Variational Autoencoder):** A reconstruction-based model ideal for unsupervised learning.
- **VAE with Noise:** A variant of the VAE model that includes noise injection (controlled by the `--noise_std` parameter) for robustness testing.
- **Diffusion Model:** A model based on diffusion processes for image generation and reconstruction.
- **VAE (Variational Autoencoder):** A reconstruction-based model ideal for unsupervised learning.
- **VAE with Noise:** A variant of the VAE model that includes noise injection (controlled by the `--noise_std` parameter) for robustness testing.
- **Enhanced VAE with Score Matching:** Extends the VAE with a latent score matching mechanism. By enabling the `--enhanced_ood` flag along with additional parameters (e.g., `--score_epochs`, `--score_lr`, `--score_noise_std`, `--ld_step_size`, `--ld_num_steps`), the model refines its latent space using denoising score matching and Langevin dynamics to improve OOD detection.

Additional model architectures can be integrated seamlessly by extending the base model class and updating the `MODEL_IDS` and model generator utilities accordingly.

---

## Usage

Before running experiments, activate the conda environment:

```bash
conda activate ood-detection
```

The main training and evaluation command is exposed via the `train-model` command. This command handles:

- Downloading and preparing datasets.
- Configuring the device (CPU or GPU).
- Creating the model based on the selected architecture.
- Setting up optimizers, schedulers, and training loops.
- Logging hyperparameters and metrics including OOD scores using the integrated ExperimentLogger.

### Command-Line Arguments

Key command-line arguments include:

- **Dataset and Partitioning:**
  - `output_path` – Directory for experiment results.
  - `dataset_path` – Directory where datasets are stored or downloaded.
  - `--in_dataset` – Name of the in-distribution dataset. For medical imaging applications, you might use:
    - `pathmnist` for a MedMNIST variant.
    - `nih_chest_xrays` when training on chest X-ray data.
  - `--ood_datasets` – List of OOD dataset IDs. For external evaluation, you might supply a list such as `tissuemnist` and `chestmnist` (or other MedMNIST Datasets).
  - `--partition_method` – Partitioning method:
    - `"internal"` reserve a subset of in-distribution classes, such as specific conditions within MedMNIST or
    - `"external"` use separate OOD datasets.
    For external partitioning, outlier datasets are pre-processed to match the channel dimensions of the inlier dataset by selecting one channel or replicating channels.
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

#### Training a Variational Autoencoder (VAE) for a Medical Imaging Task

For an in-distribution experiment using the PathMNIST with external OOD evaluation on complementary Chest- and TissueMNIST datasets:

```bash
python -m ood_detection ood-pipeline \
  /path/to/output \
  /path/to/dataset \
  --in_dataset pathmnist \
  --ood_datasets tissuemnist chestmnist \
  --model_type vae \
  --epochs 10 \
  --batchsize 64 \
  --learning_rate 0.001 \
  --latent_dim 100 \
  --use_gpu
```

This command will:
- Download/configure the MedMNIST (`pathmnist`) dataset.
- Pre-process external OOD datasets (`tissuemnist` and `chestmnist`) to match the inlier channel dimensions (via channel selection or replication).
- Instantiate a VAE with a 100-dimensional latent space.
- Train the model for 10 epochs.
- Log training and OOD evaluation metrics.

#### Training with Internal Partitioning

For an internal partitioning scenario—where a subset of in-distribution classes (e.g., specific conditions within MedMNIST) is reserved—the command is similar, with the addition of specifying the number of inlier classes:

```bash
python -m ood_detection ood-pipeline \
  /path/to/output \
  /path/to/dataset \
  --in_dataset pathmnist \
  --partition_method internal \
  --num_inliers 5 \
  --model_type vae \
  --epochs 10 \
  --batchsize 64 \
  --learning_rate 0.001 \
  --latent_dim 100 \
  --use_gpu
```

#### OOD Detection on Chest X-rays

When working with a chest X-ray dataset (e.g., `nih_chest_xrays`), the framework supports evaluation across different imaging views. For example:

```bash
python -m ood_detection ood-pipeline \
  /path/to/output \
  /path/to/dataset \
  --in_dataset nih_chest_xrays \
  --ood_datasets different_view_set \
  --model_type vae \
  --epochs 10 \
  --batchsize 32 \
  --learning_rate 0.001 \
  --latent_dim 100 \
  --use_gpu
```

This command demonstrates that for chest X-rays, OOD evaluation can leverage varying imaging perspectives to assess model robustness.

---

## Pipeline Details

The core training logic is implemented in the `OODPipelineCommand` and its descriptor:

- **Dataset Preparation:**
  Downloads and prepares the in-distribution dataset; creates training and validation loaders. For medical imaging, this includes handling datasets such as MedMNIST (`pathmnist`) and chest X-ray collections.

- **Model and Device Configuration:**
  Sets up the model (e.g., VAE) and configures the training device based on the `--use_gpu` flag.

- **Optimizer and Scheduler:**
  Configures the optimizer (SGD, Adam, etc.) and learning rate schedulers based on command-line parameters.

- **Training and OOD Evaluation:**
  After training, the pipeline partitions data either internally (selecting a subset of inlier classes) or externally (using dedicated OOD datasets). For external partitioning, outlier datasets are adjusted to match the channel dimensions of the inlier dataset—by either using one channel or replicating channels—ensuring compatibility. These scores are then used to compute and log OOD metrics.

- **Experiment Logging:**
  The integrated ExperimentLogger saves hyperparameters, logs per-epoch metrics to CSV (and TensorBoard if enabled), and logs OOD evaluation details into partition-specific CSV files.

---

## Experiment Logging

The `ExperimentLogger` class provides detailed logging for every experiment:

- **Hyperparameter Logging:**
  Saves all key parameters (including the experiment start timestamp) to a YAML file.

- **Metric Tracking:**
  Logs training metrics (loss, accuracy) and OOD metrics (computed anomaly scores) to CSV files. Optionally, metrics are also sent to TensorBoard for visualization.

- **Image Logging:**
  For reconstruction-based models, both original and reconstructed images are logged via TensorBoard or saved as files for later inspection.

This logging mechanism ensures that every experiment is fully reproducible and that all details are stored for future reference.

---

## Contributing

Contributions for additional commands, new model architectures, or improvements to experiment logging are welcome. Please ensure that any new features align with the framework’s focus on reproducibility and comprehensive experiment tracking.

---

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

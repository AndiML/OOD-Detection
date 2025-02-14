
# OOD-Detection Framework

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python Version](https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg)](https://www.python.org/downloads/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
![Contributors](https://img.shields.io/github/contributors/AndiML/Action-Recognition-System.svg)

## Overview

The **OOD-Detection** framework automates Out-Of-Distribution (OOD) detection by automatically downloading the necessary datasets for both training and inference. This framework simplifies the process of implementing OOD detection in your projects by providing a ready-to-use setup for data handling, model training, and evaluation.


## Table of Contents
- [OOD-Detection Framework](#ood-detection-framework)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)

## Installation

To install the OOD-Detection framework, clone the repository and create the conda environment using the provided `environment.yaml` file:

```bash
git clone git@github.com:AndiML/OOD-Detection.git
cd OOD-Detection
conda env create -f environment.yaml
```

If you install new packages later, make sure to update the environment file with:

```bash
conda env export | grep -v "prefix" | grep -v "numpy" > environment.yaml
```

## Usage

Before running the project, activate the conda environment:

```bash
conda activate ood-detection
```

Then run the main module with your desired arguments:

```bash
python -m action-recognition <arguments...>
```

After you are done, deactivate the environment:

```bash
conda deactivate
```

# Latent Space Optimization using Diffusion Models

## Thesis Overview
This thesis investigates latent space optimization (LSO) using diffusion models, aiming to scale black-box optimization to larger generators and high-resolution images. We explore how to maximize a quantifiable attribute (e.g., smiling) while preserving realism and faithfulness. We compare three search spaces: (i) Stable Diffusion autoencoder latents, (ii) a compact learned latent (LatentVQVAE over SD latents), and (iii) LoRA space optimization (LoRASO), which optimizes low-dimensional adapter embeddings. Under matched settings, LoRASO achieves the strongest attribute gains and realism, highlighting the effectiveness of conditioning space optimization over traditional latent space methods.

## Repository Description
Official code repository for the Master's thesis _Latent Space Optimization using Diffusion Models_, submitted to the Data and Web Science Group (Prof. Dr.-Ing. Margret Keuper) at the University of Mannheim.

## Setup Guide

### Environment
```bash
conda create -n optdif1 python=3.12
conda activate optdif1
pip install -r requirements.txt
```

### Data Import
The FFHQ dataset is too large to be included in this repository. The `images1024x1024` version of the FFHQ dataset can be downloaded from [here](https://drive.google.com/drive/folders/1ucUww4h_7dmn_Q0JJRSqSreV9hT2bZTs?usp=drive_link). Copy the Google Drive folder `data` to the root directory of the workspace, and unzip the image archive. The directory structure should look like this:
```
├── data
│   └── ffhq
│       ├── images1024x1024
│       │   ├── 00000.png
│       │   ├── ...
│       │   ├── 69999.png
│       ├── ffhq-dataset-v2.json
│       ├── smile_scores.json
│       └── smile_scores_scaled.json
```

### Classifier Import
The CelebA classifier can be downloaded from [here](https://drive.google.com/drive/folders/1ucUww4h_7dmn_Q0JJRSqSreV9hT2bZTs?usp=drive_link). Copy the Google Drive folder `models` to the root directory of the workspace.

## Model Training
This repository provides scripts for training various autoencoder models and finetuning the Stable Diffusion VAE or other components. The model implementations can be found under `src/models/`, and their training scripts are located in `src/run/`. Training can be started by adapting and running the appropriate Slurm scripts under `slurm/train/`.

## Optimization
The primary work of this thesis is to perform optimization within autoencoder latent spaces or alternative embedding spaces. For optimization, we rely on direct gradient-based optimization (`src/gbo/`) or a Bayesian optimization framework (`src/bo/`). We consider three main approaches to optimization and their implementation can be found directly in `src/lso_<approach>.py` and launched via the respective Slurm script in `slurm/lso/`:
1. **Optimization in Stable Diffusion Latent Space**: Direct optimization in a Stable Diffusion autoencoder latent space.
2. **Optimization in LatentVQVAE Latent Space**: Optimization in a compact learned latent space, exemplified by a LatentVQVAE that further encodes Stable Diffusion (SD) latents.
3. **Optimization in LoRA Conditioning Space**: This approach builds on the [CTRLorALTer](https://compvis.github.io/LoRAdapter/) paper and exploits intermediate representations of a LoRAdapter for optimization.

## Credits
- Thanks to the author of [Latent Space Optimization via Weighted Retraining of Deep Generative Models with Application on Image Data](https://github.com/janschwedhelm/master-thesis) for implementations of latent space optimization upon which parts of this repository are based.
- Thanks to the authors of [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the autoencoder implementation reused here.
- Thanks to the authors of [CTRLorALTer](https://github.com/CompVis/LoRAdapter) for providing their LoRAdapter implementation.
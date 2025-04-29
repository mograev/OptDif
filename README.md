# Latent Space Optimization using Diffusion Models

## Setup Guide

### Environment Setup
1. Create conda environment:
    `conda create -n optdif1 python=3.12`
2. Activate conda environment
    `conda activate optdif1`
3. Install dependencies
    `pip install -r requirements.txt`


### Data Import
The FFHQ dataset is too large to be included in this repository. The images1024x1024 version of the FFHQ dataset can be downloaded from [here](https://drive.google.com/drive/folders/1ucUww4h_7dmn_Q0JJRSqSreV9hT2bZTs?usp=drive_link)

4. Move FFHQ images archive to `data/ffhq/` directory and unzip it
   `unzip ffhq-dataset.zip -d data/ffhq/`

5. Move the json file `ffhq-dataset-v2.json` to `data/ffhq/` directory
   The directory structure should look like this:
   ```
    ├── data
    │   └── ffhq
    │       ├── images1024x1024
    │       │   ├── 00000.png
    │       │   ├── ...
    │       │   ├── 69999.png
    │       ├── ffhq_smile_scores.json
    │       ├── ffhq_smile_scores_scaled.json
    │       └── ffhq-dataset-v2.json
    ´´´

6. Run the script `slurm/data/preprocessing.sh` to preprocess the dataset. This will create a new directory `data/ffhq/pt_images`.


### Latent Model Training
7. Adapt the partition in `slurm/train/latent_vae_ffhq.sh`.

8. Run the script `slurm/train/latent_vae_ffhq.sh` to train the model.
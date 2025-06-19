""" Run weighted retraining for FFHQ with the VAE of the Stable Diffusion model and a LatentVAE """

import argparse
from pathlib import Path
import os
import sys
import logging
import subprocess
import time
from tqdm.auto import tqdm
import yaml
import pickle

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import pytorch_lightning as pl
import numpy as np
from torchvision import transforms

# Stable Diffusion VAE
from diffusers import AutoencoderKL

# My imports
from src.dataloader.ffhq import FFHQWeightedDataset
from src.dataloader.utils import SimpleFilenameToTensorDataset
from src.dataloader.weighting import DataWeighter
from src.classification.smile_classifier import SmileClassifier
from src.models.latent_models import LatentVQVAE, LatentVQVAE2
from src.utils import SubmissivePlProgressbar
from src import DNGO_TRAIN_FILE, GP_TRAIN_FILE, BO_OPT_FILE, GBO_TRAIN_FILE, GBO_OPT_FILE, ENTMOOT_TRAIN_OPT_FILE, GBO_FI_TRAIN_FILE, GBO_FI_OPT_FILE


# Weighted Retraining arguments
def add_wr_args(parser): 
    """ Add arguments for weighted retraining """

    wr_group = parser.add_argument_group("Weighted Retraining")
    wr_group.add_argument("--seed", type=int, required=True)
    wr_group.add_argument("--query_budget", type=int, required=True)
    wr_group.add_argument("--retraining_frequency", type=int, required=True)
    wr_group.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    wr_group.add_argument("--result_path", type=str, required=True, help="root directory to store results in")
    wr_group.add_argument("--sd_vae_path", type=str, default=None, help="path to pretrained Stable Diffusion VAE model to use")
    wr_group.add_argument("--latent_model_config_path", type=str, required=True, help="path to the config file of the latent model")
    wr_group.add_argument("--latent_model_ckpt_path", type=str, default=None, help="path to pretrained latent model to use")
    wr_group.add_argument("--predictor_path", type=str, default=None, help="path to pretrained predictor to use")
    wr_group.add_argument("--scaled_predictor", type=bool, default=False, help="whether the predictor is scaled")
    wr_group.add_argument("--predictor_attr_file", type=str, default=None, help="path to attribute file of the predictor")
    wr_group.add_argument("--n_retrain_epochs", type=float, default=1., help="number of epochs to retrain for")
    wr_group.add_argument("--n_init_retrain_epochs", type=float, default=None, help="None to use n_retrain_epochs, 0.0 to skip init retrain")

    return parser

# Optimization arguments
def add_opt_args(parser):
    """ Add arguments for training and optimization of surrogate model. """

    # Common arguments
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument("--opt_strategy", type=str, choices=["GBO", "DNGO", "GP", "ENTMOOT", "GBO_PCA", "GBO_FI"], help="Optimization strategy to use")
    opt_group.add_argument("--n_out", type=int, default=5, help="Number of points to return from optimization")
    opt_group.add_argument("--n_starts", type=int, default=20, help="Number of optimization runs with different initial values")
    opt_group.add_argument("--n_rand_points", type=int, default=8000, help="Number of random points to sample for surrogate model training")
    opt_group.add_argument("--n_best_points", type=int, default=2000, help="Number of best points to sample for surrogate model training")
    opt_group.add_argument("--feature_selection", type=str, default=None, choices=["PCA", "FI"], help="Feature selection method to use: 'PCA' or 'FI'. If None, no feature selection is applied.")
    opt_group.add_argument("--feature_selection_dims", type=int, default=512, help="Number of (PCA or FI) dimensions to use. If feature_selection is None, this is ignored.")
    opt_group.add_argument("--feature_selection_model_path", type=str, default=None, help="Path to the feature selection model. If feature_selection is None, this is ignored.")

    # BO arguments (used for both DNGO and GP)
    bo_group = parser.add_argument_group("BO")
    bo_group.add_argument("--n_samples", type=int, default=10000, help="Number of samples to draw from sample distribution")
    bo_group.add_argument("--sample_distribution", type=str, default="normal", help="Distribution to sample from: 'normal' or 'uniform'")
    bo_group.add_argument("--opt_method", type=str, default="SLSQP", choices=["SLSQP", "COBYLA", "L-BFGS-B"], help="Optimization method to use: 'SLSQP', 'COBYLA' 'L-BFGS-B'")
    bo_group.add_argument("--opt_constraint_threshold", type=float, default=None, help="Log-density threshold for optimization constraint")
    bo_group.add_argument("--opt_constraint_strategy", type=str, default="gmm_fit", help="Strategy for optimization constraint: only 'gmm_fit' is implemented")
    bo_group.add_argument("--n_gmm_components", type=int, default=None, help="Number of components used for GMM fitting")
    bo_group.add_argument("--sparse_out", type=bool, default=True, help="Whether to filter out duplicate outputs")

    # GP arguments
    bo_group.add_argument("--n_inducing_points", type=int, default=500, help="Number of inducing points to use for GP (if initializing)")

    return parser

# Setup logging
logger = logging.getLogger("lso-ffhq")

def setup_logger(logfile):
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


def _run_command(command, command_name):
    logger.debug(f"{command_name} command:")
    logger.debug(command)
    start_time = time.time()
    env = os.environ.copy()
    run_result = subprocess.run(command, capture_output=True, env=env)
    if run_result.returncode != 0:
        raise RuntimeError(
            f"{command_name} failed with return code {run_result.returncode}.\n"
            f"Error output:\n{run_result.stderr.decode('utf-8')}"
        )
    logger.debug(f"{command_name} done in {time.time() - start_time:.1f}s")


def _retrain_latent_model(latent_model, datamodule, save_dir, version_str, num_epochs, device):

    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)

    # Create custom saver and logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True)

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")
    
    # Enable PyTorch anomaly detection
    with torch.autograd.set_detect_anomaly(True):
        # Create trainer
        trainer = pl.Trainer(
            accelerator="gpu" if device == "cuda" else "cpu",
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=0.0,
            logger=tb_logger,
            callbacks=[checkpointer],
            enable_progress_bar=False,
        )

        # Fit model
        trainer.fit(latent_model, datamodule=datamodule)


def _choose_best_rand_points(args, dataset):
    """ Helper function to choose points for training surrogate model """
    chosen_point_set = set()

    # Best scores at start
    targets_argsort = np.argsort(-dataset.attr_train.flatten())
    for i in range(args.n_best_points):
        chosen_point_set.add(targets_argsort[i])

    # Random points
    candidate_rand_points = np.random.choice(
        len(targets_argsort),
        size=args.n_rand_points + args.n_best_points,
        replace=False,
    )
    for i in candidate_rand_points:
        if i not in chosen_point_set and len(chosen_point_set) < (
            args.n_rand_points + args.n_best_points
        ):
            chosen_point_set.add(i)
    assert len(chosen_point_set) == (args.n_rand_points + args.n_best_points)
    chosen_points = sorted(list(chosen_point_set))

    return chosen_points


def _encode_images(sd_vae, dataloader, device):
    """ Helper function to encode images into SD-VAE latent space """
    z_encode = []

    # Move VAE to the correct device
    sd_vae = sd_vae.to(device)
    
    with torch.no_grad():
        for image_tensor_batch in dataloader:
            # Move images to the correct device
            images = image_tensor_batch.to(device)

            # Encode images into latent space
            latents = sd_vae.encode(images).latent_dist.sample()  # Use the sampled latent distribution
            
            # Append each sampled latent to the list
            for latent in latents:
                z_encode.append(latent.cpu())

    # Free up GPU memory
    sd_vae = sd_vae.cpu()
    torch.cuda.empty_cache()

    return z_encode

def _encode_latents(latent_model, dataloader, opt_strategy, device):
    """ Helper function to encode SD latents into lower-dimensional latent space """
    zz_encode = []

    # Move VAE to the correct device
    latent_model = latent_model.to(device)

    with torch.no_grad():
        for sd_tensor_batch in dataloader:
            # Move sd latents to the correct device
            sd_latents = sd_tensor_batch.to(device)

            # Encode sd latents into lower dim latent space
            if opt_strategy == "ENTMOOT":
                # Get discrete indices for ENTMOOT
                latents = latent_model.encode(sd_latents)[2]
            elif isinstance(latent_model, LatentVQVAE2):
                latents_b, latents_t, _, _ = latent_model.encode(sd_latents)
                # Store latent shapes in model
                latent_model.latent_b_shape = latents_b.shape[1:]
                latent_model.latent_t_shape = latents_t.shape[1:]
                latent_model.divide_point = int(torch.tensor(latent_model.latent_b_shape).prod().item())
                # Flatten the two parts
                latents_b = latents_b.view(latents_b.shape[0], -1)
                latents_t = latents_t.view(latents_t.shape[0], -1)
                # Concatenate the two parts
                latents = torch.cat([latents_b, latents_t], dim=1)
            else:
                latents = latent_model.encode(sd_latents)[0]

            zz_encode.append(latents.cpu().numpy())

    # Free up GPU memory
    latent_model = latent_model.cpu()
    torch.cuda.empty_cache()

    # Concatenate all points and convert to numpy
    zz_encode = np.concatenate(zz_encode, axis=0)

    return zz_encode


def _decode_and_predict(latent_model, sd_vae, predictor, z, opt_strategy, device):
    """ Helper function to decode VAE latent vectors and calculate their properties """
    # Decode all points in a fixed decoding radius
    decoded_images = []
    sd_latents = []
    batch_size = 1000

    # Move VAE to the correct device
    latent_model = latent_model.to(device)
    sd_vae = sd_vae.to(device)

    with torch.no_grad():
        for j in range(0, len(z), batch_size):
            # Move latent vectors to the correct device
            latents = z[j: j + batch_size].to(device)

            # Decode latent vectors to SD latents
            if opt_strategy == "ENTMOOT":
                # Decode discrete indices for ENTMOOT
                sd_lats = latent_model.decode_code(latents)
            elif isinstance(latent_model, LatentVQVAE2):
                # Decode latent vectors for LatentVQVAE2
                latents_b = latents[:, :latent_model.divide_point].view(
                    -1, *latent_model.latent_b_shape
                )
                latents_t = latents[:, latent_model.divide_point:].view(
                    -1, *latent_model.latent_t_shape
                )
                sd_lats = latent_model.decode(latents_b, latents_t)
            else:
                sd_lats = latent_model.decode(latents)

            # Decode SD latents to images
            dec_imgs = sd_vae.decode(sd_lats).sample
            dec_imgs = dec_imgs.cpu()  # Move to CPU for further processing
            
            # Append decoded images and SD latents to the list
            decoded_images.append(dec_imgs)
            sd_latents.append(sd_lats.cpu())

    # Free up GPU memory
    latent_model = latent_model.cpu()
    sd_vae = sd_vae.cpu()
    torch.cuda.empty_cache()

    # Concatenate all points and convert to numpy
    decoded_images = torch.cat(decoded_images, dim=0).to(device)
    sd_latents = torch.cat(sd_latents, dim=0)

    # Normalize decoded points
    img_mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    img_std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    decoded_images_normalized = (decoded_images - img_mean) / img_std

    # Calculate objective function values and choose which points to keep
    predictions = predictor(decoded_images_normalized, batch_size=1000)

    return decoded_images.cpu(), predictions, sd_latents


def latent_optimization(args, latent_model, sd_vae, predictor, datamodule, num_queries_to_do, data_file, run_folder, device="cpu", pbar=None, postfix=None):
    """ Perform latent space optimization using traditional local optimization strategies """

    # -- Prepare Optimization ------------------------------------- #
    logger.debug("Preparing Optimization")

    # First, choose BO points to train!
    chosen_indices = _choose_best_rand_points(args, datamodule)

    # Create a new dataset with only the chosen points
    filenames = [datamodule.data_train[i] for i in chosen_indices]
    sd_latent_dir = datamodule.mode_dirs["sd_latent"]
    temp_dataset = SimpleFilenameToTensorDataset(filenames, sd_latent_dir)
    targets = datamodule.attr_train[chosen_indices]

    # Create a dataloader for the chosen points
    temp_dataloader = DataLoader(
        temp_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Encode the data to the lower-dimensional latent space
    latent_points = _encode_latents(latent_model, temp_dataloader, args.opt_strategy, args.device)
    logger.debug(f"Latent points shape: {latent_points.shape}")

    # Save points to file
    if args.opt_strategy in ["GP", "DNGO", "ENTMOOT"]:
        targets = -targets.reshape(-1, 1)  # Since it is a minimization problem
    else:
        targets = targets.reshape(-1, 1)

    # Save the file
    np.savez_compressed(
        data_file,
        X_train=latent_points.astype(np.float64),
        y_train=targets.astype(np.float64),
    )

    # Save old progress bar description
    if pbar is not None:
        old_desc = pbar.desc

    iter_seed = int(np.random.randint(10000))

    # -- Optimization based on strategy --------------------------- #

    if args.opt_strategy in ["GP", "DNGO"]:

        # -- 1. Fit surrogate model ------------------------------- #

        new_bo_file = run_folder / f"bo_train_res.npz"
        log_path = run_folder / f"bo_train.log"

        if args.opt_strategy == "GP":

            gp_train_command = [
                "python",
                GP_TRAIN_FILE,
                f"--nZ={args.n_inducing_points}",
                f"--seed={iter_seed}",
                f"--data_file={str(data_file)}",
                f"--save_file={str(new_bo_file)}",
                f"--logfile={str(log_path)}",
                f"--device={args.device}",
                "--kmeans_init",
                f"--feature_selection={args.feature_selection}" if args.feature_selection else "",
                f"--feature_selection_dims={args.feature_selection_dims}" if args.feature_selection else "",
                f"--feature_selection_model_path={args.feature_selection_model_path}" if args.feature_selection_model_path else "",
            ]

            if pbar is not None:
                pbar.set_description("GP initial fit")

            _run_command(gp_train_command, f"GP train")

        elif args.opt_strategy == "DNGO":

            dngo_train_command = [
                "python",
                DNGO_TRAIN_FILE,
                f"--seed={iter_seed}",
                f"--data_file={str(data_file)}",
                f"--save_file={str(new_bo_file)}",
                f"--logfile={str(log_path)}",
                f"--feature_selection={args.feature_selection}" if args.feature_selection else "",
                f"--feature_selection_dims={args.feature_selection_dims}" if args.feature_selection else "",
                f"--feature_selection_model_path={args.feature_selection_model_path}" if args.feature_selection_model_path else "",
                f"--device={args.device}",
            ]

            if pbar is not None:
                pbar.set_description("DNGO initial fit")

            _run_command(dngo_train_command, f"DNGO train")

        curr_bo_file = new_bo_file

        # -- 2. Optimize surrogate acquisition function ----------- #
        
        opt_path = run_folder / f"bo_opt_res.npz"
        log_path = run_folder / f"bo_opt.log"

        bo_opt_command = [
            "python",
            BO_OPT_FILE,
            f"--seed={iter_seed}",
            f"--surrogate_type={args.opt_strategy}",
            f"--surrogate_file={str(curr_bo_file)}",
            f"--data_file={str(data_file)}",
            f"--save_file={str(opt_path)}",
            f"--logfile={str(log_path)}",
            f"--sample_distribution={str(args.sample_distribution)}",
            f"--n_out={str(num_queries_to_do)}",
            f"--n_starts={args.n_starts}",
            f"--n_samples={str(args.n_samples)}",
            f"--opt_method={args.opt_method}",
            f"--sparse_out={args.sparse_out}"
        ]

        if args.opt_constraint_threshold is not None:
            bo_opt_command.append(f"--opt_constraint_threshold={args.opt_constraint_threshold}")
            bo_opt_command.append(f"--opt_constraint_strategy={args.opt_constraint_strategy}")
            bo_opt_command.append(f"--n_gmm_components={args.n_gmm_components}")

        if args.feature_selection is not None:
            bo_opt_command.append(f"--feature_selection={args.feature_selection}")
            bo_opt_command.append(f"--feature_selection_dims={args.feature_selection_dims}")

        if pbar is not None:
            pbar.set_description("optimizing acq func")
        _run_command(bo_opt_command, f"Surrogate opt")

    elif args.opt_strategy == "GBO":

        # -- 1. Fit surrogate model ------------------------------- #

        new_gbo_file = run_folder / f"gbo_train_res.npz"
        log_path = run_folder / f"gbo_train.log"

        gbo_train_command = [
            "python",
            GBO_TRAIN_FILE,
            f"--seed={iter_seed}",
            f"--data_file={str(data_file)}",
            f"--save_file={str(new_gbo_file)}",
            f"--logfile={str(log_path)}",
            f"--device={args.device}",
            f"--normalize_input",
            f"--feature_selection={args.feature_selection}" if args.feature_selection else "",
            f"--feature_selection_dims={args.feature_selection_dims}" if args.feature_selection else "",
            f"--feature_selection_model_path={args.feature_selection_model_path}" if args.feature_selection_model_path else "",
        ]

        if pbar is not None:
            pbar.set_description("GBO initial fit")

        _run_command(gbo_train_command, f"GBO train")

        curr_gbo_file = new_gbo_file

        # -- 2. Optimize surrogate acquisition function ----------- #

        opt_path = run_folder / f"gbo_opt_res.npz"
        log_path = run_folder / f"gbo_opt.log"

        gbo_opt_command = [
            "python",
            GBO_OPT_FILE,
            f"--seed={iter_seed}",
            f"--model_file={str(curr_gbo_file)}",
            f"--save_file={str(opt_path)}",
            f"--data_file={str(data_file)}",
            f"--logfile={str(log_path)}",
            f"--n_starts={args.n_starts}",
            f"--n_out={str(num_queries_to_do)}",
            f"--sample_distribution={args.sample_distribution}",
            f"--feature_selection={args.feature_selection}" if args.feature_selection else "",
            f"--feature_selection_dims={args.feature_selection_dims}" if args.feature_selection else "",
        ]      

        if pbar is not None:
            pbar.set_description("gradient-based optimization")

        _run_command(gbo_opt_command, f"GBO opt")

    elif args.opt_strategy == "ENTMOOT":

        # Fitting and optimizing surrogate model (in one script)
       new_entmoot_file = run_folder / f"entmoot_opt_res.npy"
       log_path = run_folder / f"entmoot_train_opt.log"
       opt_path = run_folder / f"entmoot_opt_res.npy"

       entmoot_train_opt_command = [
            "python",
            ENTMOOT_TRAIN_OPT_FILE,
            f"--seed={iter_seed}",
            f"--data_file={str(data_file)}",
            f"--save_file={str(new_entmoot_file)}",
            f"--predictor_path={str(args.predictor_path)}",
            f"--scaled_predictor={str(args.scaled_predictor)}",
            f"--predictor_attr_file={str(args.predictor_attr_file)}",
            f"--n_starts={str(args.n_starts)}",
            f"--n_out={str(num_queries_to_do)}",
            f"--n_dim={str(latent_model.hparams.ddconfig['embed_dim'])}",
            f"--n_embed={str(latent_model.hparams.ddconfig['n_embed'])}",
            f"--logfile={str(log_path)}",
       ]

       if pbar is not None:
           pbar.set_description("ENTMOOT training and optimization")

       _run_command(entmoot_train_opt_command, f"ENTMOOT train and opt")

    # Load point
    results = np.load(opt_path, allow_pickle=True)
    z_opt = results["z_opt"]
    z_init = results.get("z_init", None)

    # Decode point
    x_new, y_new, sd_opt = _decode_and_predict(
        latent_model,
        sd_vae,
        predictor,
        torch.as_tensor(z_opt, device=device),
        args.opt_strategy,
        device
    )

    if z_init is not None:
        x_new_init, _, _ = _decode_and_predict(
            latent_model,
            sd_vae,
            predictor,
            torch.as_tensor(z_init, device=device),
            args.opt_strategy,
            device
        )
    else:
        x_new_init = None

    # Reset pbar description
    if pbar is not None:
        pbar.set_description(old_desc)

        # Update best point in progress bar
        if postfix is not None:
            postfix["best"] = max(postfix["best"], float(max(y_new)))
            pbar.set_postfix(postfix)

    # Update data points
    if x_new_init is not None:
        return (x_new, y_new, z_opt, sd_opt, x_new_init)
    else:
        return x_new, y_new, z_opt, sd_opt


def main_loop(args):

    # Seeding
    pl.seed_everything(args.seed)

    # Make result directory
    result_dir = Path(args.result_path).resolve()
    result_dir.mkdir(parents=True)

    # Create subdirectories
    data_dir = result_dir / "data"
    data_dir.mkdir()
    sd_latent_dir = data_dir / "sd_latents"
    sd_latent_dir.mkdir()
    sd_latent_dir = Path("/BS/optdif/work/results/temp_latents")
    samples_dir = data_dir / "samples"
    samples_dir.mkdir()

    # Setup logging
    setup_logger(result_dir / "main.log")
    
    # Load datamodule
    datamodule = FFHQWeightedDataset(
        args,
        data_weighter=DataWeighter(args),
        transform=transforms.Compose([
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    )
    
    # Load pre-trained SD-VAE model
    if args.sd_vae_path == "stabilityai/stable-diffusion-3.5-medium":
        sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
        sd_vae.eval()
    else:
        raise NotImplementedError(args.sd_vae_path)
    
    # Freeze the SD-VAE model
    for param in sd_vae.parameters():
        param.requires_grad = False
    
    # Load latent VAE config
    with open(args.latent_model_config_path, "r") as f:
        latent_model_config = yaml.safe_load(f)

    # Initialize latent model
    latent_model = LatentVQVAE2(
        ddconfig=latent_model_config["ddconfig"],
        lossconfig=latent_model_config["lossconfig"],
        ckpt_path=args.latent_model_ckpt_path,
        sd_vae_path=args.sd_vae_path,
    )

    # Load pretrained (temperature-scaled) predictor
    predictor = SmileClassifier(
        model_path=args.predictor_path,
        attr_file=args.predictor_attr_file,
        scaled=args.scaled_predictor,
        device=args.device,
        logfile=result_dir / "predictor.log",
    )
    
    # Set up results tracking
    results = dict(
        opt_points=[],  # saves (default: 5) optimal points in the original input space for each retraining iteration
        opt_point_properties=[], # saves corresponding function evaluations
        opt_latent_points=[], # saves corresponding latent points
        opt_sd_latent_points=[], # saves corresponding sd latent points
        opt_model_version=[],
        params=str(sys.argv),
    )

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))  # default: 500/5 = 100
    postfix = dict(
        retrain_left=num_retrain, best=-float("inf"), n_train=len(datamodule.data_train)
    )

    # Save retraining hyperparameters in YAML format
    with open(result_dir / "retraining_hparams.yaml", "w") as f:
        yaml.dump(args.__dict__, f, default_flow_style=False)

    # # Encode images into SD-VAE latent space
    # sd_latents = _encode_images(
    #     sd_vae,
    #     datamodule.set_mode("img").train_dataloader(),
    #     args.device
    # )

    # # # # Save the encoded images as tensors
    # for filename, sd_latent in zip(datamodule.train_dataset.filename_list, sd_latents):
    #     torch.save(sd_latent, sd_latent_dir / f"{filename}.pt", pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Add SD latents to datamodule
    datamodule.add_mode("sd_latent", sd_latent_dir)

    # Main loop
    with tqdm(
        total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        for ret_idx in range(num_retrain):
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")

            # Decide whether to retrain the VAE
            samples_so_far = args.retraining_frequency * ret_idx

            # Retraining
            datamodule.set_mode("sd_latent")
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                # default: initial fine-tuning of pre-trained model for 1 epoch
                num_epochs = args.n_init_retrain_epochs
            if num_epochs > 0:
                retrain_dir = result_dir / "retraining"
                version = f"iter_{samples_so_far}"
                # default: run through 10% of the weighted training data in retraining epoch
                _retrain_latent_model(
                    latent_model, datamodule, retrain_dir, version, num_epochs, args.device
                )

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Optimize latent space
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )

            # Create a new directory for the optimization run
            opt_dir = result_dir / "opt" / f"iter_{samples_so_far}"
            opt_dir.mkdir(parents=True)
            opt_data_file = opt_dir / "data.npz"

            # Perform latent optimization
            lso_results = latent_optimization(
                args,
                latent_model,
                sd_vae,
                predictor,
                datamodule,
                num_queries_to_do,
                data_file=opt_data_file,
                run_folder=opt_dir,
                device=args.device,
                pbar=pbar,
                postfix=postfix,
            )

            if len(lso_results) == 5:
                x_new, y_new, z_query, sd_query, x_new_init = lso_results
            else:
                x_new, y_new, z_query, sd_query = lso_results
                x_new_init = None

            # Create a new directory for the sampled data
            curr_samples_dir = Path(samples_dir) / f"iter_{samples_so_far}"
            curr_samples_dir.mkdir()
            
            # Save the new latent points
            os.makedirs(curr_samples_dir / "latents")
            for i, z in enumerate(z_query):
                if not isinstance(z, torch.Tensor):
                    z = torch.from_numpy(z)
                torch.save(z, str(Path(curr_samples_dir) / f"latents/tensor_{i}.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)

            # Save the new sd latent points
            new_filename_list = []
            os.makedirs(curr_samples_dir / "sd_latents")
            for i, sd in enumerate(sd_query):
                if not isinstance(sd, torch.Tensor):
                    sd = torch.from_numpy(sd)
                torch.save(sd, str(Path(curr_samples_dir) / f"sd_latents/tensor_{i}.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
                new_filename_list.append(str(Path(curr_samples_dir) / f"sd_latents/tensor_{i}.pt"))
            
            # Save the new images
            os.makedirs(curr_samples_dir / "img_tensor")
            for i, x in enumerate(x_new):
                if not isinstance(x, torch.Tensor):
                    x = torch.from_numpy(x)
                torch.save(x, str(Path(curr_samples_dir) / f"img_tensor/tensor_{i}.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)

            # Save initial points if available
            if x_new_init is not None:
                os.makedirs(curr_samples_dir / "img_tensor_init")
                for i, x in enumerate(x_new_init):
                    if not isinstance(x, torch.Tensor):
                        x = torch.from_numpy(x)
                    torch.save(x, str(Path(curr_samples_dir) / f"img_tensor_init/tensor_{i}.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)

            # Append new points to dataset and adapt weighting
            datamodule.append_train_data(new_filename_list, y_new)

            # Save results
            results["opt_latent_points"] += [z for z in z_query]
            results["opt_sd_latent_points"] += [sd for sd in sd_query]
            results["opt_points"] += [x.detach().numpy() for x in x_new]
            results["opt_point_properties"] += [y for y in y_new]
            results["opt_model_version"] += [ret_idx] * len(x_new)
            np.savez_compressed(str(result_dir / "results.npz"), **results)

            # Final update of progress bar
            postfix["best"] = max(postfix["best"], float(y_new.max()))
            postfix["n_train"] = len(datamodule.data_train)
            pbar.set_postfix(postfix)
            pbar.update(n=num_queries_to_do)
    print("Weighted retraining finished; end of script")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # arguments and argument checking
    parser = argparse.ArgumentParser()
    parser = add_wr_args(parser)
    parser = add_opt_args(parser)
    parser = FFHQWeightedDataset.add_data_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    
    args = parser.parse_args()

    main_loop(args)
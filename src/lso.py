""" Run weighted retraining for FFHQ with the VAE of the Stable Diffusion model """

import argparse
from pathlib import Path
import os
import sys
import logging
import subprocess
import time
from tqdm.auto import tqdm
import json
import pickle

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

# Diffusers
from diffusers import AutoencoderKL

# My imports
from src.dataloader.ffhq import FFHQWeightedTensorDataset, SimpleFilenameToTensorDataset
from src.dataloader.weighting import DataWeighter
from src.classification.smile_classifier import SmileClassifier
from src.models.lit_vae import LitVAE
from src.utils import SubmissivePlProgressbar
from src import DNGO_TRAIN_FILE, DNGO_OPT_FILE


# Weighted Retraining arguments
def add_wr_args(parser): 
    """ Add arguments for weighted retraining """

    wr_group = parser.add_argument_group("Weighted Retraining")
    wr_group.add_argument("--seed", type=int, required=True)
    wr_group.add_argument("--query_budget", type=int, required=True)
    wr_group.add_argument("--retraining_frequency", type=int, required=True)
    wr_group.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    wr_group.add_argument("--result_path", type=str, required=True, help="root directory to store results in")
    wr_group.add_argument("--pretrained_model_path", type=str, default=None, help="path to pretrained model to use")
    wr_group.add_argument("--pretrained_predictor_path", type=str, default=None, help="path to pretrained predictor to use")
    wr_group.add_argument("--scaled_predictor_path", type=str, default=None, help="path to scaled predictor to use")
    wr_group.add_argument("--predictor_attr_file", type=str, default=None, help="path to attribute file of the predictor")
    wr_group.add_argument("--n_retrain_epochs", type=float, default=1., help="number of epochs to retrain for")
    wr_group.add_argument("--n_init_retrain_epochs", type=float, default=None, help="None to use n_retrain_epochs, 0.0 to skip init retrain")

    return parser

# DNGO arguments
def add_dngo_args(parser):
    """ Add arguments for DNGO training and optimization """

    bo_group = parser.add_argument_group("BO")
    bo_group.add_argument("--n_out", type=int, default=5)
    bo_group.add_argument("--n_starts", type=int, default=20, help="Number of optimization runs with different initial values")
    bo_group.add_argument("--n_samples", type=int, default=10000)
    bo_group.add_argument("--n_rand_points", type=int, default=8000)
    bo_group.add_argument("--n_best_points", type=int, default=2000)
    bo_group.add_argument("--sample_distribution", type=str, default="normal")
    bo_group.add_argument("--opt_method", type=str, default="SLSQP")
    bo_group.add_argument("--opt_constraint_threshold", type=float, default=None, help="Threshold for optimization constraint")
    bo_group.add_argument("--opt_constraint_strategy", type=str, default="gmm_fit")
    bo_group.add_argument("--n_gmm_components", type=int, default=None, help="Number of components used for GMM fitting")
    bo_group.add_argument("--sparse_out", type=bool, default=True)

    return parser

# Setup logging
logger = logging.getLogger("lso-ffhq-dngo")

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
    run_result = subprocess.run(command, capture_output=True)
    assert run_result.returncode == 0, run_result.stderr
    logger.debug(f"{command_name} done in {time.time() - start_time:.1f}s")


def retrain_vae(vae, datamodule, save_dir, version_str, num_epochs, device):

    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)
    train_pbar = SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val",)

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")
    
    # Create LitVAE module to allow for PL training
    vae_module = LitVAE(vae)
    
    # Enable PyTorch anomaly detection
    with torch.autograd.set_detect_anomaly(True):
        # Create trainer
        trainer = pl.Trainer(
            accelerator="gpu" if device == "cuda" else "cpu",
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=1,
            logger=tb_logger,
            callbacks=[train_pbar, checkpointer],
        )

        # Fit model
        trainer.fit(vae_module, datamodule)


def _choose_best_rand_points(args: argparse.Namespace, dataset):
    """ helper function to choose points for training surrogate model """
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


def _encode_images(vae, dataset, device):
    """ Helper function to encode images into VAE latent space """
    z_encode = []

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Move VAE to the correct device
    vae = vae.to(device)

    with torch.no_grad():
        for image_tensor_batch in dataloader:
            # Move images to the correct device
            images = image_tensor_batch.to(device)
            
            # Encode images into latent space
            latent_dist = vae.encode(images).latent_dist
            latents = latent_dist.mean  # Use the mean of the latent distribution
            z_encode.append(latents.cpu().numpy())

    # Free up GPU memory
    vae = vae.cpu()
    torch.cuda.empty_cache()

    # Aggregate array
    z_encode = np.concatenate(z_encode, axis=0)

    return z_encode

def _batch_decode_z_and_props(vae, predictor, z, device, get_props=True):
    """ Helper function to decode VAE latent vectors and calculate their properties """
    # Decode all points in a fixed decoding radius
    z_decode = []
    z_decode_upscaled = []
    batch_size = 1000

    with torch.no_grad():
        for j in range(0, len(z), batch_size):
            # Move latent vectors to the correct device
            latents = z[j: j + batch_size].to(device)

            # Decode latent vectors into images
            decoded_images = vae.decode(latents).sample  # Decode and get the reconstructed images
            decoded_images = decoded_images.cpu()  # Move to CPU for further processing

            # Upscale images to the desired size
            img_upscaled = torch.nn.functional.interpolate(
                decoded_images, size=(128, 128), mode="bicubic", align_corners=False
            )
            z_decode.append(decoded_images)
            z_decode_upscaled.append(img_upscaled)

    # Concatenate all points and convert to numpy
    z_decode_upscaled = torch.cat(z_decode_upscaled, dim=0).to(device)
    z_decode = torch.cat(z_decode, dim=0).to(device)

    # Normalize decoded points
    img_mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    img_std = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
    z_decode_normalized = (z_decode_upscaled - img_mean) / img_std

    # Calculate objective function values and choose which points to keep
    if get_props:
        with torch.no_grad():
            predictions = predictor(z_decode_normalized)
            probas_predictions = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()
            z_prop = probas_predictions @ np.array([0, 1, 2, 3, 4, 5])

    if get_props:
        return z_decode, z_prop
    else:
        return z_decode


def latent_optimization(args, vae, predictor, datamodule, num_queries_to_do, bo_data_file, bo_run_folder, device="cpu", pbar=None, postfix=None):
    """ Perform latent space optimization using traditional local optimization strategies """

    ##################################################
    # Prepare BO
    ##################################################

    # First, choose BO points to train!
    chosen_indices = _choose_best_rand_points(args, datamodule)

    # Create a new dataset with only the chosen points
    data = [datamodule.data_train[i] for i in chosen_indices]
    temp_dataset = SimpleFilenameToTensorDataset(data, args.pt_dir)
    targets = datamodule.attr_train[chosen_indices]

    # Next, encode the data to latent space
    latent_points = _encode_images(vae, temp_dataset, device)
    logger.debug(latent_points.shape)

    # Save points to file
    def _save_bo_data(latent_points, targets):

        # Prevent overfitting to bad points
        targets = -targets.reshape(-1, 1)  # Since it is a minimization problem

        # Save the file
        np.savez_compressed(
            bo_data_file,
            X_train=latent_points.astype(np.float64),
            X_test=[],
            y_train=targets.astype(np.float64),
            y_test=[],
        )

    _save_bo_data(latent_points, targets)

    # Part 1: fit surrogate model
    # ===============================
    iter_seed = int(np.random.randint(10000))

    new_bo_file = bo_run_folder / f"bo_train_res.npz"
    log_path = bo_run_folder / f"bo_train.log"
    
    dngo_train_command = [
        "python",
        DNGO_TRAIN_FILE,
        f"--seed={iter_seed}",
        f"--data_file={str(bo_data_file)}",
        f"--save_file={str(new_bo_file)}",
        f"--logfile={str(log_path)}",
    ]

    # Add commands for initial fitting
    dngo_fit_desc = "DNGO initial fit"

    # Set pbar status for user
    if pbar is not None:
        old_desc = pbar.desc
        pbar.set_description(dngo_fit_desc)

    # Run command
    _run_command(dngo_train_command, f"DNGO train")
    curr_bo_file = new_bo_file

    # Part 2: optimize surrogate acquisition func to query point
    # ===============================
    
    # Run optimization
    opt_path = bo_run_folder / f"bo_opt_res.npy"
    log_path = bo_run_folder / f"bo_opt.log"
    dngo_opt_command = [
        "python",
        DNGO_OPT_FILE,
        f"--seed={iter_seed}",
        f"--surrogate_file={str(curr_bo_file)}",
        f"--data_file={str(bo_data_file)}",
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
        dngo_opt_command.append(f"--opt_constraint_threshold={args.opt_constraint_threshold}")
        dngo_opt_command.append(f"--opt_constraint_strategy={args.opt_constraint_strategy}")
        dngo_opt_command.append(f"--n_gmm_components={args.n_gmm_components}")

    if pbar is not None:
        pbar.set_description("optimizing acq func")
    _run_command(dngo_opt_command, f"Surrogate opt")

    # Load point
    z_opt = np.load(opt_path)
    
    # Decode point
    x_new, y_new = _batch_decode_z_and_props(
        vae,
        predictor,
        torch.as_tensor(z_opt, device=device),
        device
    )

    # Reset pbar description
    if pbar is not None:
        pbar.set_description(old_desc)

        # Update best point in progress bar
        if postfix is not None:
            postfix["best"] = max(postfix["best"], float(max(y_new)))
            pbar.set_postfix(postfix)

    # Update datamodule with ALL data points
    return x_new, y_new, z_opt


def main_loop(args):

    # Seeding
    pl.seed_everything(args.seed)
    
    # Load data
    datamodule = FFHQWeightedTensorDataset(args, DataWeighter(args))
    datamodule.setup() # assignment into train/validation split is made and weights are set
    
    # Load pre-trained SD-VAE model
    if args.pretrained_model_path == "stabilityai/stable-diffusion-3.5-medium":
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
        vae.eval()
    else:
        raise NotImplementedError(args.pretrained_model_code)
    
    # Load pretrained (temperature-scaled) predictor
    predictor = SmileClassifier(
        model_path=args.pretrained_predictor_path,
        attr_file=args.predictor_attr_file,
        scaled_model_path=args.scaled_predictor_path,
        device=args.device,
    )
    
    # Set up results tracking
    results = dict(
        opt_points=[],  # saves (default: 5) optimal points in the original input space for each retraining iteration
        opt_point_properties=[], # saves corresponding function evaluations
        opt_latent_points=[], # saves corresponding latent points
        opt_model_version=[],
        params=str(sys.argv),
    )

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))  # default: 500/5 = 100
    postfix = dict(
        retrain_left=num_retrain, best=-float("inf"), n_train=len(datamodule.data_train)
    )

    # Main loop
    with tqdm(
        total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        # Make result directory
        result_dir = Path(args.result_path).resolve()
        result_dir.mkdir(parents=True)
        # Create subdirectories
        data_dir = result_dir / "data"
        data_dir.mkdir()
        setup_logger(result_dir / "log.txt")

        # Save retraining hyperparameters in JSON format
        with open(result_dir / 'retraining_hparams.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4)

        for ret_idx in range(num_retrain):
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")

            # Decide whether to retrain the VAE
            samples_so_far = args.retraining_frequency * ret_idx

            # Optionally do retraining
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                # default: initial fine-tuning of pre-trained model for 1 epoch
                num_epochs = args.n_init_retrain_epochs
            if num_epochs > 0:
                retrain_dir = result_dir / "retraining"
                version = f"retrain_{samples_so_far}"
                # default: run through 10% of the weighted training data in retraining epoch
                retrain_vae(
                    vae, datamodule, retrain_dir, version, num_epochs, args.device
                )

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Optimize latent space
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )

            # Create a new directory for the BO run
            bo_dir = result_dir / "bo" / f"iter{samples_so_far}"
            bo_dir.mkdir(parents=True)
            bo_data_file = bo_dir / "data.npz"

            # Perform latent optimization
            x_new, y_new, z_query = latent_optimization(
                args,
                vae,
                predictor,
                datamodule,
                num_queries_to_do,
                bo_data_file=bo_data_file,
                bo_run_folder=bo_dir,
                device=args.device,
                pbar=pbar,
                postfix=postfix,
            )

            # Save new tensor data
            if not os.path.exists(str(Path(data_dir) / f"sampled_data_iter{samples_so_far}")):
                os.makedirs(str(Path(data_dir) / f"sampled_data_iter{samples_so_far}"))

            new_filename_list = []
            for i, x in enumerate(x_new):
                torch.save(x, str(Path(data_dir) / f"sampled_data_iter{samples_so_far}/tensor_{i}.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
                new_filename_list.append(str(Path(data_dir) / f"sampled_data_iter{samples_so_far}/tensor_{i}.pt"))
            
            # Append new points to dataset and adapt weighting
            datamodule.append_train_data(new_filename_list, y_new)

            # Save results
            results["opt_latent_points"] += [z for z in z_query]
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
    # arguments and argument checking
    parser = argparse.ArgumentParser()
    parser = add_wr_args(parser)
    parser = add_dngo_args(parser)
    parser = FFHQWeightedTensorDataset.add_data_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    
    args = parser.parse_args()

    main_loop(args)
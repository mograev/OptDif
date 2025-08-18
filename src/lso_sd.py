"""
Run LSO for FFHQ with the VAE of the Stable Diffusion model.
"""

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
from torchvision import transforms
from torchvision.utils import save_image
import torch.multiprocessing as mp # Set the multiprocessing start method to spawn
import pytorch_lightning as pl
import numpy as np

# Stable Diffusion VAE
from diffusers import AutoencoderKL

# My imports
sys.path.append(os.getcwd()) # Ensure the src directory is in the Python path
from src.dataloader.ffhq import FFHQDataset
from src.dataloader.utils import OptEncodeDataset
from src.dataloader.weighting import DataWeighter
from src.classification.smile_classifier import SmileClassifier
from src.models.lit_vae import LitVAE
from src import DNGO_TRAIN_FILE, GP_TRAIN_FILE, BO_OPT_FILE, GBO_TRAIN_FILE, GBO_OPT_FILE


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
    wr_group.add_argument("--predictor_path", type=str, default=None, help="path to pretrained predictor to use")
    wr_group.add_argument("--scaled_predictor", action="store_true", help="whether the predictor uses temperature scaling")
    wr_group.add_argument("--predictor_attr_file", type=str, default=None, help="path to attribute file of the predictor")
    wr_group.add_argument("--n_retrain_epochs", type=float, default=1., help="number of epochs to retrain for")
    wr_group.add_argument("--n_init_retrain_epochs", type=float, default=None, help="None to use n_retrain_epochs, 0.0 to skip init retrain")

    return parser

# Optimization arguments
def add_opt_args(parser):
    """ Add arguments for training and optimization of surrogate model."""

    # Common arguments
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument("--opt_strategy", type=str, choices=["GBO", "DNGO", "GP"], help="Optimization strategy to use")
    opt_group.add_argument("--n_starts", type=int, default=20, help="Number of optimization runs with different initial values")
    opt_group.add_argument("--n_rand_points", type=int, default=8000, help="Number of random points to sample for surrogate model training")
    opt_group.add_argument("--n_best_points", type=int, default=2000, help="Number of best points to sample for surrogate model training")
    opt_group.add_argument("--sample_distribution", type=str, default="normal", choices=["normal", "train_data"], help="Distribution to sample from: 'normal' or 'train_data'")
    opt_group.add_argument("--feature_selection", type=str, default="None", choices=["PCA", "FI", "None"], help="Feature selection method to use: 'PCA' or 'FI'. If 'None', no feature selection is applied.")
    opt_group.add_argument("--feature_selection_dims", type=int, default=512, help="Number of (PCA or FI) dimensions to use. If feature_selection is None, this is ignored.")
    opt_group.add_argument("--feature_selection_model_path", type=str, default=None, help="Path to the feature selection model. If feature_selection is None, this is ignored.")

    # BO arguments (used for both DNGO and GP)
    bo_group = parser.add_argument_group("BO")
    bo_group.add_argument("--n_samples", type=int, default=10000, help="Number of samples to draw from sample distribution")
    bo_group.add_argument("--opt_method", type=str, default="SLSQP", choices=["SLSQP", "COBYLA", "L-BFGS-B", "trust-constr"], help="Optimization method to use: 'SLSQP', 'COBYLA' 'L-BFGS-B'")
    bo_group.add_argument("--opt_constraint", type=str, choices=["GMM", "None"], help="Strategy for optimization constraint: only 'GMM' is implemented")
    bo_group.add_argument("--n_gmm_components", type=int, default=None, help="Number of components used for GMM fitting")
    bo_group.add_argument("--sparse_out", type=bool, default=True, help="Whether to filter out duplicate outputs")

    # GP arguments
    bo_group.add_argument("--n_inducing_points", type=int, default=500, help="Number of inducing points to use for GP (if initializing)")

    return parser

# Setup logging
logger = logging.getLogger("lso-ffhq-sd")

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


def _retrain_vae(vae, datamodule, save_dir, version_str, num_epochs, device):

    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)

    # Create custom saver and logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True)

    # Wrap the VAE in a Lightning module
    vae_module = LitVAE(model=vae)

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
        trainer.fit(vae_module, datamodule=datamodule)


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

            # Flatten each latent, such that shape is (B, C*H*W)
            latents = latents.view(latents.shape[0], -1)

            # Append each sampled latent to the list
            for latent in latents:
                z_encode.append(latent.cpu().numpy())

    # Free up GPU memory
    sd_vae = sd_vae.cpu()
    torch.cuda.empty_cache()

    # Concatenate all points and convert to numpy
    z_encode = np.stack(z_encode, axis=0)

    return z_encode


def _decode_and_predict(sd_vae, predictor, z, device):
    """ Helper function to decode VAE latent vectors and calculate their properties """
    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1000

    # Move VAE to the correct device
    sd_vae = sd_vae.to(device)

    with torch.no_grad():
        for j in range(0, len(z), batch_size):
            # Move latent vectors to the correct device
            latents = z[j: j + batch_size].to(device)

            # Reshape latents to match the VAE's expected input shape (B, C', H', W')
            latents = latents.view(latents.shape[0], *sd_vae.latent_shape)

            # Decode SD latents to images
            decoded_images = sd_vae.decode(latents).sample
            decoded_images = decoded_images.cpu()  # Move to CPU for further processing

            z_decode.append(decoded_images)

    # Free up GPU memory
    sd_vae = sd_vae.cpu()
    torch.cuda.empty_cache()

    # Concatenate all points and convert to range [0, 1]
    z_decode = (torch.cat(z_decode, dim=0).to(device) + 1) / 2

    # Calculate objective function values and choose which points to keep
    predictions = predictor(z_decode, batch_size=1000)

    return z_decode.cpu(), predictions


def latent_optimization(args, sd_vae, predictor, datamodule, num_queries_to_do, data_file, run_folder, device="cpu", pbar=None, postfix=None):
    """ Perform latent space optimization using traditional local optimization strategies """

    # -- Prepare Optimization ------------------------------------- #
    logger.debug("Preparing Optimization")

    # First, choose points to train!
    chosen_indices = _choose_best_rand_points(args, datamodule)

    # Create a new dataset with only the chosen points
    temp_dataset = OptEncodeDataset(
        filename_list=[datamodule.data_train[i] for i in chosen_indices],
        img_dir=datamodule.img_dir,
        transform=datamodule.transform,
        device=datamodule.device
    )
    temp_targets = datamodule.attr_train[chosen_indices]

    # Create a dataloader for the chosen points
    temp_dataloader = DataLoader(
        temp_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Encode the data to the lower-dimensional latent space
    latent_points = _encode_images(sd_vae, temp_dataloader, args.device)
    logger.debug(f"Latent points shape: {latent_points.shape}")

    # Save points to file
    if args.opt_strategy in ["GP", "DNGO", "DNGO_PCA"]:
        targets = -temp_targets.reshape(-1, 1)  # Since it is a minimization problem
    else:
        targets = temp_targets.reshape(-1, 1)

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
    logger.debug(f"Iteration seed: {iter_seed}")

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
            ]

            # Append feature selection arguments if specified
            if args.feature_selection != "None":
                gp_train_command.append(f"--feature_selection={args.feature_selection}")
                gp_train_command.append(f"--feature_selection_dims={args.feature_selection_dims}")
                if args.feature_selection_model_path is not None:
                    gp_train_command.append(f"--feature_selection_model_path={args.feature_selection_model_path}")

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
                f"--device={args.device}",
                f"--normalize_input",
                f"--normalize_output",
            ]

            # Append feature selection arguments if specified
            if args.feature_selection != "None":
                dngo_train_command.append(f"--feature_selection={args.feature_selection}")
                dngo_train_command.append(f"--feature_selection_dims={args.feature_selection_dims}")
                if args.feature_selection_model_path is not None:
                    dngo_train_command.append(f"--feature_selection_model_path={args.feature_selection_model_path}")

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

        if args.opt_constraint != "None":
            bo_opt_command.append(f"--opt_constraint={args.opt_constraint}")
            bo_opt_command.append(f"--n_gmm_components={args.n_gmm_components}")

        if args.feature_selection != "None":
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
        ]

        # Append feature selection arguments if specified
        if args.feature_selection != "None":
            gbo_train_command.append(f"--feature_selection={args.feature_selection}")
            gbo_train_command.append(f"--feature_selection_dims={args.feature_selection_dims}")
            if args.feature_selection_model_path is not None:
                gbo_train_command.append(f"--feature_selection_model_path={args.feature_selection_model_path}")

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
        ]

        if args.feature_selection != "None":
            gbo_opt_command.append(f"--feature_selection={args.feature_selection}")
            gbo_opt_command.append(f"--feature_selection_dims={args.feature_selection_dims}")

        if pbar is not None:
            pbar.set_description("gradient-based optimization")

        _run_command(gbo_opt_command, f"GBO opt")

    # Delete data and train results files to save space
    if os.path.exists(data_file):
        os.remove(data_file)
    if args.opt_strategy in ["GP", "DNGO"]:
        if os.path.exists(curr_bo_file):
            os.remove(curr_bo_file)
    elif args.opt_strategy == "GBO":
        if os.path.exists(curr_gbo_file):
            os.remove(curr_gbo_file)

    # Load point (and init points if available)
    results = np.load(opt_path, allow_pickle=True)
    z_opt = results["z_opt"]
    z_init = results.get("z_init", None)
    z_indices = results.get("z_indices", None)

    # Derive original images from initial indices
    # Only applicable if initial points are sampled from the training data
    x_orig = None
    if args.sample_distribution == "train_data" and z_indices is not None:
        x_orig = [temp_dataset[int(idx)] for idx in z_indices]
        y_orig = [temp_targets[int(idx)] for idx in z_indices]

    # Decode point
    x_opt, y_opt = _decode_and_predict(
        sd_vae,
        predictor,
        torch.as_tensor(z_opt, device=device),
        device
    )

    # Decode initial points if available
    x_init = None
    if z_init is not None:
        x_init, y_init = _decode_and_predict(
            sd_vae,
            predictor,
            torch.as_tensor(z_init, device=device),
            device
        )

    # Reset pbar description
    if pbar is not None:
        pbar.set_description(old_desc)

        # Update best point in progress bar
        if postfix is not None:
            postfix["best"] = max(postfix["best"], float(max(y_opt)))
            pbar.set_postfix(postfix)

    # Return data points
    if x_init is not None and x_orig is not None:
        return (x_opt, y_opt, z_opt, x_init, y_init, x_orig, y_orig)
    elif x_init is not None:
        return (x_opt, y_opt, z_opt, x_init, y_init)
    else:
        return (x_opt, y_opt, z_opt)


def main_loop(args):

    # Seeding
    pl.seed_everything(args.seed)

    # Make result directory
    result_dir = Path(args.result_path).resolve()
    result_dir.mkdir(parents=True)

    # Create subdirectories
    data_dir = result_dir / "data"
    data_dir.mkdir()
    samples_dir = data_dir / "samples"
    samples_dir.mkdir()

    # Setup logging
    setup_logger(result_dir / "main.log")

    # Load pre-trained SD-VAE model
    if args.sd_vae_path == "stabilityai/stable-diffusion-3.5-medium":
        sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="vae")
        sd_vae.eval()
    elif args.sd_vae_path == "stable-diffusion-v1-5/stable-diffusion-v1-5":
        sd_vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
        sd_vae.eval()
    else:
        try:
            sd_vae = AutoencoderKL.from_pretrained(args.sd_vae_path)
        except Exception as e:
            logger.error(f"Failed to load SD-VAE from {args.sd_vae_path}: {e}")

    # Obtain shape of the latent space
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 256, 256)
        latent_shape = sd_vae.encode(dummy).latent_dist.sample().shape[1:]  # (C', H', W')
    sd_vae.latent_shape = latent_shape

    # Load datamodule
    datamodule = FFHQDataset(
        args,
        data_weighter=DataWeighter(args),
        encoder=sd_vae,
        transform=transforms.Compose([
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    )

    # Load pretrained (temperature-scaled) predictor
    predictor = SmileClassifier(
        model_path=args.predictor_path,
        attr_file=args.predictor_attr_file,
        scaled=args.scaled_predictor,
        device=args.device,
    )

    # Set up results tracking
    results = dict(
        opt_point_properties=[], # saves corresponding function evaluations
        init_point_properties=[], # saves initial point properties if available
        orig_point_properties=[], # saves original point properties if available
        opt_model_version=[],
        params=str(sys.argv),
    )

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))  # default: 500/5 = 100
    postfix = dict(
        retrain_left=num_retrain, best=-float("inf"), n_train=len(datamodule.data_train)
    )

    # Save retraining hyperparameters in YAML format
    with open(result_dir / "hparams.yaml", "w") as f:
        yaml.dump(args.__dict__, f, default_flow_style=False)

    # Main loop
    with tqdm(
        total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:

        for ret_idx in range(num_retrain):
            # Log global time
            logger.info(f"Retraining iteration {ret_idx + 1}/{num_retrain} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

            # Update progress bar
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")

            # Decide whether to retrain the VAE
            samples_so_far = args.retraining_frequency * ret_idx

            # Retraining
            datamodule.set_encode(False)
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                # default: initial fine-tuning of pre-trained model for 1 epoch
                num_epochs = args.n_init_retrain_epochs
            if num_epochs > 0:
                retrain_dir = result_dir / "retraining"
                version = f"retrain_{samples_so_far}"
                # default: run through 10% of the weighted training data in retraining epoch
                _retrain_vae(
                    sd_vae, datamodule, retrain_dir, version, num_epochs, args.device
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

            # Unpack results based on the number of returned values
            if len(lso_results) == 7:
                x_opt, y_opt, z_opt, x_init, y_init, x_orig, y_orig = lso_results
            elif len(lso_results) == 5:
                x_opt, y_opt, z_opt, x_init, y_init = lso_results
                x_orig, y_orig = None, None
            else:
                x_opt, y_opt, z_opt = lso_results
                x_init, y_init = None, None
                x_orig, y_orig = None, None

            # Create a new directory for the sampled data
            curr_samples_dir = Path(samples_dir) / f"iter_{samples_so_far}"
            curr_samples_dir.mkdir()

            # Save the new latent points
            os.makedirs(curr_samples_dir / "latent")
            for i, z in enumerate(z_opt):
                if not isinstance(z, torch.Tensor):
                    z = torch.from_numpy(z)
                torch.save(z, str(Path(curr_samples_dir) / f"latent/{i}.pt"))

            # Save the new images
            os.makedirs(curr_samples_dir / "img_opt")
            new_filename_list = []
            for i, x in enumerate(x_opt):
                if not isinstance(x, torch.Tensor):
                    x = torch.from_numpy(x)
                img_path = str(Path(curr_samples_dir) / f"img_opt/{i}.png")
                save_image(x, img_path, normalize=True)
                new_filename_list.append(img_path)

            # Save initial points (reconstructions) if available
            if x_init is not None:
                os.makedirs(curr_samples_dir / "img_init")
                for i, x in enumerate(x_init):
                    if not isinstance(x, torch.Tensor):
                        x = torch.from_numpy(x)
                    img_path = str(Path(curr_samples_dir) / f"img_init/{i}.png")
                    save_image(x, img_path, normalize=True)

            # Save original images if available
            if x_orig is not None:
                os.makedirs(curr_samples_dir / "img_orig")
                for i, x in enumerate(x_orig):
                    if not isinstance(x, torch.Tensor):
                        x = torch.from_numpy(x)
                    img_path = str(Path(curr_samples_dir) / f"img_orig/{i}.png")
                    save_image(x, img_path, normalize=True)

            # Append new points to dataset and adapt weighting
            datamodule.append_train_data(new_filename_list, y_opt)

            # Save results
            results["opt_point_properties"] += [float(y) for y in y_opt]
            results["init_point_properties"] += [float(y) for y in y_init] if y_init is not None else []
            results["orig_point_properties"] += [float(y) for y in y_orig] if y_orig is not None else []
            results["opt_model_version"] += [ret_idx] * len(x_opt)
            np.savez_compressed(str(result_dir / "results.npz"), **results)

            # Final update of progress bar
            postfix["best"] = max(postfix["best"], float(y_opt.max()))
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
    parser = FFHQDataset.add_data_args(parser)
    parser = DataWeighter.add_weight_args(parser)

    args = parser.parse_args()

    main_loop(args)
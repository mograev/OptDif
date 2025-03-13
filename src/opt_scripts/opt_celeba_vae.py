""" Run weighted retraining for CelebA with the VAE model """

import argparse
from pathlib import Path
import os
import sys
import logging
import subprocess
import time
from tqdm.auto import tqdm, trange

import pytorch_lightning as pl
import ruamel.yaml

# My imports
from src.dataloader_celeba_weighting import *
from src.celeba.celeba_vae import CelebaVAE
from src.temperature_scaling import *
from src.resnet50 import resnet50
from src import utils
from src.opt_scripts import base as wr_base
from src import GP_TRAIN_FILE, GP_OPT_FILE, DNGO_TRAIN_FILE

logger = logging.getLogger("celeba-opt")


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


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu):

    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)
    train_pbar = utils.SubmissivePlProgressbar(process_position=1)

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

    # Create trainer, default batch size: 32
    trainer = pl.Trainer(
        gpus=1 if gpu else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=checkpointer,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar],
    )

    # Fit model
    trainer.fit(model, datamodule)


def _choose_best_rand_points(args: argparse.Namespace, dataset):
    """ helper function to choose points for training surrogate model """
    chosen_point_set = set()

    # Best scores at start
    targets_argsort = np.argsort(-dataset.prop_train.flatten())
    for i in range(args.n_best_points):
        chosen_point_set.add(targets_argsort[i])
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


def _encode_images(model, dataset):
    """ helper function to encode images to latent space """
    z_encode = []
    batch_size = 128

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    for image_tensor_batch in dataloader:
        with torch.no_grad():
            img = model.encode_to_latent_space(image_tensor_batch[0].to(model.device))
            img = img.cpu().numpy()
            z_encode.append(img)
            del img

    # Aggregate array
    z_encode = np.concatenate(z_encode, axis=0)
    return z_encode


def _batch_decode_z_and_props(model, predictor, z, get_props=True):
    """
    helper function to decode some latent vectors and calculate their properties
    """
    # Decode all points in a fixed decoding radius
    z_decode = []
    z_decode_upscaled = [] 
    batch_size = 1000
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            img = model.decode_deterministic(z[j: j + batch_size])
            # img = img.cpu().numpy()
            img_upscaled = torch.nn.functional.interpolate(img, size=(128, 128), mode='bicubic',
                                                           align_corners=False)
            z_decode.append(img.cpu())                                               
            z_decode_upscaled.append(img_upscaled.cpu())
            del img

    # Concatentate all points and convert to numpy
    z_decode_upscaled = torch.cat(z_decode_upscaled, dim=0).to(model.device)
    z_decode = torch.cat(z_decode, dim=0).to(model.device)
    # z_decode = np.around(z_decode)  # convert to int
    # z_decode = z_decode[:, 0, ...]  # Correct slicing

    # normalize decoded points
    img_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(model.device)
    img_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(model.device)
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


def latent_optimization(args, model, scaled_predictor, datamodule, num_queries_to_do, bo_data_file, bo_run_folder, pbar=None, postfix=None):
    """ perform latent space optimization using traditional local optimization strategies """

    ##################################################
    # Prepare BO
    ##################################################

    # First, choose BO points to train!
    chosen_indices = _choose_best_rand_points(args, datamodule)

    data = [datamodule.data_train[i] for i in chosen_indices]
    temp_dataset = SimpleFilenameToTensorDataset(data, args.tensor_dir)

    targets = datamodule.prop_train[chosen_indices]

    # Next, encode the data to latent space
    if args.gpu:
        model = model.cuda()
    latent_points = _encode_images(model, temp_dataset)

    model = model.cpu()  # Make sure to free up GPU memory
    torch.cuda.empty_cache()  # Free the memory up for tensorflow

    # Save points to file
    def _save_bo_data(latent_points, targets):

        # Prevent overfitting to bad points
        #targets = np.maximum(targets, args.invalid_score)
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
    
    if args.bo_surrogate == "GP":
        gp_train_command = [
            "python",
            GP_TRAIN_FILE,
            f"--nZ={args.n_inducing_points}",
            f"--seed={iter_seed}",
            f"--data_file={str(bo_data_file)}",
            f"--save_file={str(new_bo_file)}",
            f"--logfile={str(log_path)}",
        ]

        # Add commands for initial fitting
        gp_fit_desc = "GP initial fit"
        gp_train_command += [
            "--init",
            "--kmeans_init",
        ]

        # Set pbar status for user
        if pbar is not None:
            old_desc = pbar.desc
            pbar.set_description(gp_fit_desc)

        # Run command
        _run_command(gp_train_command, f"GP train")
        curr_bo_file = new_bo_file
    elif args.bo_surrogate == "DNGO":
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
    else:
        raise NotImplementedError(args.bo_surrogate)

    # Part 2: optimize surrogate acquisition func to query point
    # ===============================
    
    # Run GP opt script
    opt_path = bo_run_folder / f"bo_opt_res.npy"
    log_path = bo_run_folder / f"bo_opt.log"
    if args.opt_constraint_threshold:
        bo_opt_command = [
            "python",
            GP_OPT_FILE,
            f"--seed={iter_seed}",
            f"--surrogate_file={str(curr_bo_file)}",
            f"--data_file={str(bo_data_file)}",
            f"--save_file={str(opt_path)}",
            f"--logfile={str(log_path)}",
            f"--n_samples={str(args.n_samples)}",
            f"--sample_distribution={str(args.sample_distribution)}",
            f"--n_out={str(num_queries_to_do)}",
            f"--bo_surrogate={args.bo_surrogate}",
            f"--n_starts={args.n_starts}",
            f"--opt_method={args.opt_method}",
            f"--sparse_out={args.sparse_out}",
            f"--opt_method={args.opt_method}",
            f"--pretrained_model_type={args.pretrained_model_type}",
            f"--opt_constraint_threshold={args.opt_constraint_threshold}",
            f"--opt_constraint_strategy={args.opt_constraint_strategy}",
            f"--n_gmm_components={args.n_gmm_components}",
        ]
    else:
        bo_opt_command = [
            "python",
            GP_OPT_FILE,
            f"--seed={iter_seed}",
            f"--surrogate_file={str(curr_bo_file)}",
            f"--data_file={str(bo_data_file)}",
            f"--save_file={str(opt_path)}",
            f"--logfile={str(log_path)}",
            f"--n_samples={str(args.n_samples)}",
            f"--sample_distribution={str(args.sample_distribution)}",
            f"--n_out={str(num_queries_to_do)}",
            f"--bo_surrogate={args.bo_surrogate}",
            f"--n_starts={args.n_starts}",
            f"--opt_method={args.opt_method}",
            f"--sparse_out={args.sparse_out}",
            f"--opt_method={args.opt_method}",
            f"--pretrained_model_type={args.pretrained_model_type}",
        ]
    if pbar is not None:
        pbar.set_description("optimizing acq func")
    _run_command(bo_opt_command, f"Surrogate opt")

    # Load point
    z_opt = np.load(opt_path)
    
    # Decode point
    x_new, y_new = _batch_decode_z_and_props(
        model,
        scaled_predictor,
        torch.as_tensor(z_opt, device=model.device)
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
    datamodule = CelebaWeightedTensorDataset(args, utils.DataWeighter(args))
    datamodule.setup("fit") # assignment into train/validation/split is made and weights are set

    # Load pre-trained VAEmodel
    if args.pretrained_model_type == "vae":
        model = CelebaVAE.load_from_checkpoint(args.pretrained_model_file)
        model.beta = model.hparams.beta_final  # Override any beta annealing
    else:
        raise NotImplementedError(args.pretrained_model_type)

    # Load pre-trained VAE model
    if args.pretrained_model_type == "vae":
        model = CelebaVAE.load_from_checkpoint(args.pretrained_model_file)
        model.beta = model.hparams.beta_final  # Override any beta annealing
    else:
        raise NotImplementedError(args.pretrained_model_type)

    # Load pretrained (temperature-scaled) CelebA-Dialog predictor
    checkpoint_predictor = torch.load(args.pretrained_predictor_file)
    predictor = resnet50(attr_file=args.attr_file)
    predictor.load_state_dict(checkpoint_predictor['state_dict'], strict=True)
    predictor.eval()

    state_dict_scaled_predictor = torch.load(args.scaled_predictor_state_dict)
    scaled_predictor = ModelWithTemperature(predictor, 3)
    scaled_predictor.load_state_dict(state_dict_scaled_predictor, strict=True)
    scaled_predictor.to(model.device)
    scaled_predictor.eval()

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
        result_dir = Path(args.result_root).resolve()
        result_dir.mkdir(parents=True)
        data_dir = result_dir / "data"
        data_dir.mkdir()
        setup_logger(result_dir / "log.txt")

        # Save retraining hyperparameters
        with open(result_dir / 'retraining_hparams.yaml', 'w') as f:
            ruamel.yaml.dump(args.__dict__, f, default_flow_style=False)

        for ret_idx in range(num_retrain):
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")

            # Decide whether to retrain
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
                retrain_model(
                    model, datamodule, retrain_dir, version, num_epochs, args.gpu
                )

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Do querying!
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )
            if args.lso_strategy == "opt":
                gp_dir = result_dir / "gp" / f"iter{samples_so_far}"
                gp_dir.mkdir(parents=True)
                bo_data_file = gp_dir / "data.npz"
                x_new, y_new, z_query = latent_optimization(
                    args,
                    model,
                    scaled_predictor,
                    datamodule,
                    num_queries_to_do,
                    bo_data_file=bo_data_file,
                    bo_run_folder=gp_dir,
                    pbar=pbar,
                    postfix=postfix,
                )
            else:
                raise NotImplementedError(args.lso_strategy)

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
    parser = CelebaWeightedTensorDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    parser = wr_base.add_common_args(parser)
    parser = wr_base.add_gp_args(parser)

    args = parser.parse_args()

    main_loop(args)

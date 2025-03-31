"""Calculates the Frechet Inception Distance (FID) to evalulate sample quality
Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/janschwedhelm/master-thesis
"""

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import TensorDataset
import pytorch_lightning as pl

from tqdm import tqdm
import argparse
import pickle
import time
from pathlib import Path

from src.dataloader.ffhq import FFHQWeightedTensorDataset
from src.dataloader.weighting import DataWeighter
from src.models.inception import InceptionV3

NUM_WORKERS = 2
DIMS = 2048


def add_fid_args(parser):
    """
    Adds arguments for FID calculation to the argument parser.
    Args:
        parser: The argument parser to add arguments to.
    Returns:
        The updated argument parser.
    """
    fid_group = parser.add_argument_group(title="fid")
    fid_group.add_argument("--seed", type=int, required=True)
    fid_group.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    fid_group.add_argument("--result_dir", type=str, required=True, help="directory to store results in")
    fid_group.add_argument("--sample_path", type=str, required=True, help="a .npz file with 'opt_points' key containing the test samples")

    return parser


def get_activations(data, model, batch_size=50, dims=2048, device='cpu', num_workers=8):
    """
    Calculates the activations of the pool_3 layer for all images.
    Params:
    -- data        : Either an np.ndarray that contains image data, a tensor or a datamodule
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        if batch_size > data.shape[0]:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = data.shape[0]

        data = torch.as_tensor(data, dtype=torch.float)

        if data.ndim < 4:
            data = data.unsqueeze(1) # insert dimension for number of channels in image
        tensor_data = TensorDataset(data)

        dataloader = torch.utils.data.DataLoader(tensor_data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=num_workers)

        pred_arr = np.empty((data.shape[0], dims))
    else:
        dataloader = data
        pred_arr = np.empty((data.dataset.__len__(), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch[0].to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(data, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=2):
    """Calculation of the statistics used by the FID.
    Params:
    -- data        : Either an np.ndarray that contains image data or a datamodule
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(data, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_given_tensors(data1, data2, batch_size, device, dims, num_workers=2, is_one_channel=False):
    """ Calculates the FID between two sets of images """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3(output_blocks=[block_idx], is_one_channel=is_one_channel).to(device)

    m1, s1 = calculate_activation_statistics(data1, model, batch_size,
                                             dims, device, num_workers)
    m2, s2 = calculate_activation_statistics(data2, model, batch_size,
                                             dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main(args):
    # Seeding
    pl.seed_everything(args.seed)

    # Make result directory
    result_dir = Path(args.result_dir).resolve()

    # Load samples
    with np.load(args.sample_path) as npz:
        samples = npz["opt_points"]

    # Load dataset
    datamodule = FFHQWeightedTensorDataset(args, DataWeighter(args))
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()

    print("Start FID computation")

    start_time = time.time()
    fid_score = calculate_fid_given_tensors(samples, dataloader, args.batch_size,
                                            args.device, dims=DIMS, num_workers=NUM_WORKERS)
    time_needed = time.time() - start_time
    results = dict(test_set=args.sample_path,
                   fid_score=fid_score,
                   time=time_needed)

    print(f"Resulting FID score: {fid_score}")
    print(f"Time needed: {time_needed} seconds")

    # Save results
    with open(result_dir, "wb") as f:
        pickle.dump(results, f)

    print("Computation finished; end of script")


if __name__ == "__main__":
    # arguments and argument checking
    parser = argparse.ArgumentParser()

    parser = FFHQWeightedTensorDataset.add_data_args(parser)
    parser = DataWeighter.add_weight_args(parser)
    parser = add_fid_args(parser)

    args = parser.parse_args()

    main(args)
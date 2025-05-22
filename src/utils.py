""" Code for various helper functions and classes """
import sys
import gzip
import pickle
import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import pytorch_lightning as pl
from scipy.stats import multivariate_normal


def zero_mean_unit_var_normalization(X, mean=None, std=None):

    compute_mean_std = mean is None and std is None

    if compute_mean_std:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    if compute_mean_std:
        return X_normalized, mean, std
    else:
        return X_normalized


def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean

# Various pytorch functions
def _get_zero_grad_tensor(device):
    """ return a zero tensor that requires grad. """
    loss = torch.as_tensor(0.0, device=device)
    loss = loss.requires_grad_(True)
    return loss


def save_object(obj, filename):
    """ Function that saves an object to a file using pickle """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, "wb") as dest:
        dest.write(result)
    dest.close()


def print_flush(text):
    print(text)
    sys.stdout.flush()


def update_hparams(hparams, model):

    # Make the hyperparameters match
    for k in model.hparams.keys():
        try:
            if vars(hparams)[k] != model.hparams[k]:
                print(
                    f"Overriding hparam {k} from {model.hparams[k]} to {vars(hparams)[k]}"
                )
                model.hparams[k] = vars(hparams)[k]
        except KeyError:  # not all keys match, it's ok
            pass

    # Add any new hyperparameters
    for k in vars(hparams).keys():
        if k not in model.hparams.keys():
            print(f'Adding missing hparam {k} with value "{vars(hparams)[k]}".')
            model.hparams[k] = vars(hparams)[k]


def add_default_trainer_args(parser, default_root=None):
    pl_trainer_grp = parser.add_argument_group("pl trainer")
    pl_trainer_grp.add_argument("--gpu", action="store_true")
    pl_trainer_grp.add_argument("--seed", type=int, default=0)
    pl_trainer_grp.add_argument("--root_dir", type=str, default=default_root)
    pl_trainer_grp.add_argument("--load_from_checkpoint", type=str, default=None)
    pl_trainer_grp.add_argument("--max_epochs", type=int, default=1000)


# functions needed for various utilities
def log_gmm_density(x, mu, variances):
    """
    Computes logarithm of density of a Gaussian mixture model (GMM) with diagonal covariance matrix.
    :param x: Quantiles, with the last axis of x denoting the components.
    :param mu: Means of GMM components.
    :param variances: Variances of GMM components.
    :return: Log-likelihood of quantiles under the specified GMM.
    """
    densities = np.array([multivariate_normal.pdf(x, mean=mu[i], cov=np.diag(variances[i])) for i in range(len(mu))])
    return np.log(1/x.shape[0]) + np.log(np.mean(densities.T, axis=1))


def sparse_subset(points, r):
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r.

    """
    result = []
    index_list = []
    for i, p in enumerate(points):
        if all(np.linalg.norm(p-q) >= r for q in result):
            result.append(p)
            index_list.append(i)
    return np.array(result), index_list


def output_to_label(output):
    """
    INPUT
    - output: [num_attr, batch_size, num_classes]
    OUTPUT
    - scores: [num_attr, batch_size, num_classes] (softmaxed)
    - label: [num_attr, batch_size]
    """
    scores = []
    labels = []
    for attr_idx in range(len(output)):
        _, label = torch.max(input=output[attr_idx], dim=1)
        label = label.cpu().numpy()[0]
        labels.append(label)

        score_per_attr = output[attr_idx].cpu().numpy()[0]
        # softmax
        score_per_attr = (np.exp(score_per_attr) /
                          np.sum(np.exp(score_per_attr)))
        scores.append(score_per_attr)

    scores = torch.FloatTensor(scores)
    labels = torch.LongTensor(labels)

    return labels, scores


def load_image_predictor(img_path,
                         transform=transforms.Compose([transforms.ToTensor()
                                                       ])):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.to(device).unsqueeze(0)

    img_mean = torch.Tensor([0.485, 0.456,
                             0.406]).view(1, 3, 1, 1).to(device)
    img_std = torch.Tensor([0.229, 0.224,
                            0.225]).view(1, 3, 1, 1).to(device)
    image = (image - img_mean) / img_std

    return image


class SubmissivePlProgressbar(pl.callbacks.ProgressBar):
    """ progress bar with tqdm set to leave """

    def __init__(self, process_position: int):
        super().__init__()
        self.process_position = process_position

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Retraining Progress",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar
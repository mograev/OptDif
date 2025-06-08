"""
FID Score calculation for real images and generated or reconstructed images.

Source: https://github.com/steffen-jung/SpectralGAN/blob/main/FID/FIDScore.py
"""

from src.models.inception import InceptionV3
import numpy as np
import os
import torch
from scipy import linalg

################################################################
class FIDScore:
    """FID Score Calculator"""
    
    f_cached_stats_real = "inception.stats_real.{}.cache"
    
    ############################################################
    def __init__(self, img_size, device="cpu", batch_size=32, num_workers=4):
        """
        Initialize the FIDScore class.
        Args:
            img_size (int): The size of the images (assumed square).
            device (str): The device to use for computation (e.g., "cpu" or "cuda").
            batch_size (int): The batch size for processing images.
            num_workers (int): The number of workers for the DataLoader.
        """
        self.device = device
        print(f"Using device for FID: {self.device}")
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.model = InceptionV3(
            resize_input = True,
            normalize_input = False
        )

        self.model.eval()
        self.model.to(device)
        
        if os.path.isfile(FIDScore.f_cached_stats_real.format(img_size)):
            print(f"Loading cached stats for FID: {FIDScore.f_cached_stats_real.format(img_size)}")
            self.mu_real, self.sigma_real = torch.load(
                FIDScore.f_cached_stats_real.format(img_size),
                map_location = device,
                weights_only=False
            )
            self.is_fitted = True
        else:
            self.is_fitted = False
    
    ############################################################
    def fit(self, data):
        """
        Compute the mean and covariance of the features extracted from the data using the InceptionV3 model.
        Args:
            data (torch.utils.data.Dataset): The dataset to compute the statistics from.
        Returns:
            mu (numpy.ndarray): The mean of the features.
            sigma (numpy.ndarray): The covariance of the features.
        """
            
        data_size = len(data)
        dims = 2048

        # Create a dataloader for the data
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        # Initialize an array to store the predictions, shaped (data_size, dims)
        pred_arr = np.empty((data_size, dims))
        i_arr = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Get predictions from the model
            pred = self.model(batch)[0].cpu().numpy()
            
            # Get the number of predictions in the batch
            pred_len = pred.shape[0]
            
            # Store predictions in the array
            pred_arr[i_arr:i_arr+pred_len, :] = pred[:, :, 0, 0]
            i_arr += pred_len

        # Compute mean and covariance of the predictions
        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        
        return mu, sigma
    
    ############################################################
    def fit_real(self, data_real):
        """
        Fit the model to the real data and save the statistics.
        Args:
            data_real (torch.utils.data.Dataset): The real dataset to compute the statistics from.
        """
        if self.is_fitted:
            print("FID already fitted, skipping.")
            return
        else:
            print("FID not fitted, computing real stats...")
            # Fit the model to the real data
            self.mu_real, self.sigma_real = self.fit(data_real)

            # Save the real stats to a file
            torch.save(
                (self.mu_real, self.sigma_real),
                FIDScore.f_cached_stats_real.format(self.img_size)
            )
            self.is_fitted = True
        
    ############################################################
    def compute_score_from_data(self, data_fake, eps=1E-6):
        """
        Compute the FID score between the real and fake data.
        Args:
            data_fake (torch.utils.data.Dataset): The fake dataset to compute the statistics from.
            eps (float): A small value to avoid numerical issues.
        Returns:
            score (float): The FID score.
        """
        mu_fake, sigma_fake = self.fit(data_fake)
        return self.compute_score(mu_fake, sigma_fake, eps)
        
    ############################################################
    def compute_score(self, mu2, sigma2, eps=1E-6):
        """
        Compute the FID score between the real and fake data.
        Args:
            mu2 (numpy.ndarray): The mean of the fake data.
            sigma2 (numpy.ndarray): The covariance of the fake data.
            eps (float): A small value to avoid numerical issues.
        Returns:
            score (float): The FID score.
        """
        mu1, sigma1 = self.get_real_stats()

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
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1E-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
    
        tr_covmean = np.trace(covmean)
    
        return ( diff.dot(diff) +
                 np.trace(sigma1) +
                 np.trace(sigma2) -
                 2 * tr_covmean )
    
    ############################################################
    def get_real_stats(self):
        """
        Get the real statistics (mean and covariance) of the dataset.
        Returns:
            mu_real (numpy.ndarray): The mean of the real data.
            sigma_real (numpy.ndarray): The covariance of the real data.
        """
        return self.mu_real, self.sigma_real
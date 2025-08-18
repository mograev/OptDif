"""
The Spectral metric compares the radial power-spectrum statistics of two image
distributions via L1, following the style of metrics/fid.py.
Source (adapted from): https://github.com/steffen-jung/SpectralGAN/blob/main/SpectralLoss.py
"""

import os
import torch
import numpy as np

################################################################
class SpectralScore:
    """Spectral Score Calculator"""

    f_cached_stats_real = "spectral.stats_real.{}.cache"

    ############################################################
    def __init__(self, img_size, device="cpu", batch_size=32, num_workers=4):
        """
        Initialize the SpectralScore class.
        Args:
            img_size (int): The size of the images (assumed square).
            device (str): The device to use for computation (e.g., "cpu" or "cuda").
            batch_size (int): The batch size for processing images.
            num_workers (int): The number of workers for the DataLoader.
        """
        self.device   = device
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Using device for SpectralScore: {self.device}")

        # Pre‑compute radial masks for one‑sided FFT
        rows = cols = img_size
        shift_rows = rows // 2
        cols_onesided = cols // 2 + 1

        r = np.indices((rows, cols_onesided)) - np.array([[[shift_rows]], [[0]]])
        r = np.sqrt(r[0, :, :] ** 2 + r[1, :, :] ** 2).astype(int)
        r = np.fft.ifftshift(r, axes=0)           # put frequency 0 top‑left

        r_max = int(r.max())
        self.vector_length = r_max + 1

        r_torch = torch.from_numpy(r).long()
        mask = (r_torch.unsqueeze(0) == torch.arange(r_max + 1).view(-1, 1, 1)).to(torch.float32)
        mask = mask.unsqueeze(0)                  # (1, R, H, W/2+1)
        mask_sum = mask.sum(dim=(2, 3), keepdim=False)  # (1, R)

        # store on chosen device
        self.mask = mask.to(device)
        self.mask_n = (1.0 / mask_sum).to(torch.float32).to(device)

        # load cached “real” statistics if available
        if os.path.isfile(self.f_cached_stats_real.format(img_size)):
            print(f"Loading cached stats for SpectralScore: "
                  f"{self.f_cached_stats_real.format(img_size)}")
            self.mu_real = torch.load(
                self.f_cached_stats_real.format(img_size),
                map_location=device,
                weights_only=False
            )
            self.is_fitted = True
        else:
            self.is_fitted = False

    ############################################################
    def fit(self, data, eps=1e-8):
        """
        Compute the mean radial spectrum for a dataset.
        Args:
            data (torch.utils.data.Dataset): The dataset to compute the statistics from.
            eps (float): A small value to avoid division by zero.
        Returns:
            mu (numpy.ndarray): The mean radial spectrum.
        """
        dims       = self.vector_length
        data_size  = len(data)
        pred_arr   = np.empty((data_size, dims), dtype=np.float64)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        i_arr = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=True)

                # convert RGB -> grayscale if necessary
                if batch.shape[1] == 3:
                    batch = (0.299 * batch[:, 0, :, :] +
                             0.587 * batch[:, 1, :, :] +
                             0.114 * batch[:, 2, :, :])

                # 2‑D real FFT, one‑sided
                fft_res = torch.fft.rfft2(batch, norm='ortho')
                fft_abs = fft_res.real ** 2 + fft_res.imag ** 2
                fft_abs = fft_abs + eps
                fft_abs = 20 * torch.log10(fft_abs)

                # build radial profiles
                fft_rep = fft_abs.unsqueeze(1).expand(-1, self.vector_length, -1, -1)
                profile = (fft_rep * self.mask).sum((2, 3)) * self.mask_n

                # normalise profile to [0,1] per image
                profile = profile - profile.min(dim=1, keepdim=True)[0]
                profile = profile / (profile.max(dim=1, keepdim=True)[0] + eps)

                pred_len = profile.shape[0]
                pred_arr[i_arr:i_arr + pred_len, :] = profile.cpu().numpy()
                i_arr += pred_len

        mu = np.mean(pred_arr, axis=0)
        return mu

    ############################################################
    def fit_real(self, data_real, eps=1e-8):
        """
        Fit the model to the real data and save the statistics.
        Args:
            data_real (torch.utils.data.Dataset): The real dataset to compute the statistics from.
            eps (float): A small value to avoid division by zero.
        """
        if self.is_fitted:
            print("SpectralScore already fitted, skipping.")
            return

        print("Fitting SpectralScore on real data...")
        self.mu_real = self.fit(data_real, eps=eps)

        torch.save(
            self.mu_real,
            self.f_cached_stats_real.format(self.img_size)
        )
        self.is_fitted = True

    ############################################################
    def compute_score_from_data(self, data_fake, eps=1e-8):
        """
        Compute the Spectral Score between the real and fake data.
        Args:
            data_fake (torch.utils.data.Dataset): The fake dataset to compute the statistics from.
            eps (float): A small value to avoid division by zero.
        Returns:
            score (float): The Spectral Score.
        """
        mu_fake = self.fit(data_fake, eps=eps)
        return self.compute_score(mu_fake)

    ############################################################
    def compute_score(self, mu_fake):
        """
        Compute the Spectral Score between the real and fake data.
        Args:
            mu_fake (numpy.ndarray): The mean radial spectrum of the fake data.
        Returns:
            score (float): The Spectral Score.
        """
        mu_real = self.get_real_stats()
        # Compute the L1 distance between the real and fake statistics
        return float(np.abs(mu_real - mu_fake).sum())

    ############################################################
    def get_real_stats(self):
        """
        Get the real statistics.
        Returns:
            mu_real (numpy.ndarray): The mean radial spectrum of the real data.
        """
        if not self.is_fitted:
            raise RuntimeError("Real statistics not fitted yet.")
        return self.mu_real
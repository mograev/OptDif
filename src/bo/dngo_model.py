"""
Deep Networks for Global Optimization. This module performs bayesian linear regression with basis function
extracted from a feed forward neural network.
Reference: [1] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish,
            N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
            Scalable Bayesian Optimization Using Deep Neural Networks
            Proc. of ICML'15
Code Sources:
- https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/dngo.py
- https://github.com/janschwedhelm/master-thesis/blob/main/src/dngo/base_model.py
"""

import time
import logging
import numpy as np
import emcee

import torch
import torch.nn as nn
import torch.optim as optim
from scipy import optimize

from src.utils import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from src.bo.bayesian_linear_regression import BayesianLinearRegression, Prior


class Net(nn.Module):
    """
    Feed forward neural network with 3 hidden layers and tanh activation function.
    """
    def __init__(self, n_inputs, n_units=[50, 50, 50]):
        """
        Initializes the neural network.
        Args:
            n_inputs (int): Number of input features
            n_units (list): Number of units in each hidden layer
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc3 = nn.Linear(n_units[1], n_units[2])
        self.out = nn.Linear(n_units[2], 1)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return self.out(x)

    def basis_funcs(self, x):
        """
        Computes the basis functions for the input data.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        return x


class DNGO:
    """Deep Networks for Global Optimization (DNGO) model."""

    def __init__(self, batch_size=10, num_epochs=50,
                 learning_rate=0.01,
                 adapt_epoch=5000, n_units_1=50, n_units_2=50, n_units_3=50,
                 alpha=1.0, beta=1000, prior=None, do_mcmc=True,
                 n_hypers=20, chain_length=2000, burnin_steps=2000,
                 normalize_input=True, normalize_output=True, rng=None):
        """
        Initializes the DNGO model.
        Args:
            batch_size (int): Batch size for training the neural network
            num_epochs (int): Number of epochs for training
            learning_rate (float): Initial learning rate for Adam
            adapt_epoch (int): Defines after how many epochs the learning rate will be decayed by a factor 10
            n_units_1 (int): Number of units in layer 1
            n_units_2 (int): Number of units in layer 2
            n_units_3 (int): Number of units in layer 3
            alpha (float): Hyperparameter of the Bayesian linear regression
            beta (float): Hyperparameter of the Bayesian linear regression
            prior (Prior object): Prior for alpa and beta. If set to None the default prior is used
            do_mcmc (bool): If set to true different values for alpha and beta are sampled via MCMC from the marginal log likelihood
                            Otherwise the marginal log likehood is optimized with scipy fmin function
            n_hypers (int): Number of samples for alpha and beta
            chain_length (int): The chain length of the MCMC sampler
            burnin_steps (int): The number of burnin steps before the sampling procedure starts
            normalize_output (bool): Zero mean unit variance normalization of the output values
            normalize_input (bool): Zero mean unit variance normalization of the input values
            rng (np.random.RandomState): Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.X = None
        self.y = None
        self.network = None
        self.alpha = alpha
        self.beta = beta
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        logging.info("Normalize input: %s" % self.normalize_input)
        logging.info("Normalize output: %s" % self.normalize_output)

        # MCMC hyperparameters
        self.do_mcmc = do_mcmc
        self.n_hypers = n_hypers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        if prior is None:
            self.prior = Prior(rng=self.rng)
        else:
            self.prior = prior

        # Network hyper parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3
        self.adapt_epoch = adapt_epoch
        self.network = None
        self.models = []
        self.hypers = None

    def train(self, X, y, do_optimize=True):
        """
        Trains the model on the provided data.
        Args:
            X (np.ndarray): Input data points. The dimensionality of X is (N, D),
                with N as the number of points and D is the number of input dimensions.
            y (np.ndarray): The corresponding target values of the input data points.
            do_optimize (bool): If True, optimize hyperparameters, otherwise sample them
        """
        # Basic shape cheks
        assert X.shape[0] == y.shape[0], "Number of inputs and targets must match."
        assert len(X.shape) == 2 and len(y.shape) == 1, "Incorrect shape for X or y."

        start_time = time.time()

        # Normalize inputs
        if self.normalize_input:
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X
            
        # Normalize ouputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        self.y = self.y[:, None]

        # Check if we have enough points to create a minibatch otherwise use all data points
        if self.X.shape[0] <= self.batch_size:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size

        # Create the neural network
        features = X.shape[1]

        self.network = Net(n_inputs=features, n_units=[self.n_units_1, self.n_units_2, self.n_units_3])

        optimizer = optim.Adam(self.network.parameters(),
                               lr=self.init_learning_rate)

        # Start training
        lc = np.zeros([self.num_epochs])
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X, self.y,
                                                  batch_size, shuffle=True):
                inputs = torch.Tensor(batch[0])
                targets = torch.Tensor(batch[1])

                optimizer.zero_grad()
                output = self.network(inputs)
                loss = torch.nn.functional.mse_loss(output, targets)
                loss.backward()
                optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
            logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))

        # Design matrix
        self.Theta = self.network.basis_funcs(torch.Tensor(self.X)).data.numpy()

        if do_optimize:
            if self.do_mcmc:
                self.sampler = emcee.EnsembleSampler(self.n_hypers, 2,
                                                     self.marginal_log_likelihood)

                # Do a burn-in in the first iteration
                if not self.burned:
                    # Initialize the walkers by sampling from the prior
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                    # Run MCMC sampling
                    result = self.sampler.run_mcmc(self.p0,
                                                   self.burnin_steps,
                                                   rstate0=self.rng)
                    self.p0 = result.coords

                    self.burned = True

                # Start sampling
                pos = self.sampler.run_mcmc(self.p0,
                                            self.chain_length,
                                            rstate0=self.rng)

                # Save the current position, it will be the startpoint in
                # the next iteration
                self.p0 = pos.coords

                # Take the last samples from each walker set them back on a linear scale
                linear_theta = np.exp(self.sampler.chain[:, -1])
                self.hypers = linear_theta
                self.hypers[:, 1] = 1 / self.hypers[:, 1]
            else:
                # Optimize hyperparameters of the Bayesian linear regression
                p0 = self.prior.sample_from_prior(n_samples=1)
                res = optimize.fmin(self.negative_mll, p0)
                self.hypers = [[np.exp(res[0]), 1 / np.exp(res[1])]]
        else:

            self.hypers = [[self.alpha, self.beta]]

        logging.info("Hypers: %s" % self.hypers)
        self.models = []
        for sample in self.hypers:
            # Instantiate a model for each hyperparameter configuration
            model = BayesianLinearRegression(alpha=sample[0],
                                             beta=sample[1],
                                             basis_func=None)
            model.train(self.Theta, self.y[:, 0], do_optimize=False)

            self.models.append(model)

    def marginal_log_likelihood(self, theta):
        """
        Log likelihood of the data marginalised over the weights w. See chapter 3.5 of
        the book by Bishop of an derivation.
        Args:
            theta (np.ndarray): Hyperparameters of the Bayesian linear regression
        Returns:
            float: The marginal log likelihood
        """
        if np.any(theta == np.inf):
            return -np.inf

        if np.any((-10 > theta) + (theta > 10)):
            return -np.inf

        alpha = np.exp(theta[0])
        beta = 1 / np.exp(theta[1])

        D = self.Theta.shape[1]
        N = self.Theta.shape[0]

        K = beta * np.dot(self.Theta.T, self.Theta)
        K += np.eye(self.Theta.shape[1]) * alpha
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.linalg.LinAlgError:
             K_inv = np.linalg.inv(K + np.random.rand(K.shape[0], K.shape[1]) * 1e-8)

        m = beta * np.dot(K_inv, self.Theta.T)
        m = np.dot(m, self.y)

        mll = D / 2 * np.log(alpha)
        mll += N / 2 * np.log(beta)
        mll -= N / 2 * np.log(2 * np.pi)
        mll -= beta / 2. * np.linalg.norm(self.y - np.dot(self.Theta, m), 2)
        mll -= alpha / 2. * np.dot(m.T, m)
        mll -= 0.5 * np.log(np.linalg.det(K) + 1e-10)

        if np.any(np.isnan(mll)):
            return -1e25
        return mll

    def negative_mll(self, theta):
        """
        Returns the negative marginal log likelihood (for optimizing it with scipy).
        Args:
            theta (np.ndarray): Hyperparameters of the Bayesian linear regression
        Returns:
            float: The negative marginal log likelihood
        """
        nll = -self.marginal_log_likelihood(theta)
        return nll

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        """
        Iterates over the training data in mini-batches of size batchsize.
        Args:
            inputs (np.ndarray): Input data points
            targets (np.ndarray): Target values
            batchsize (int): Size of each mini-batch
            shuffle (bool): If True, shuffle the data before creating mini-batches
        Yields:
            np.ndarray: Mini-batch of input data points
            np.ndarray: Mini-batch of target values
        """
        # Check if the number of training points is the same as the number of targets
        assert inputs.shape[0] == targets.shape[0], \
            "The number of training points is not the same"
        if shuffle:
            indices = np.arange(inputs.shape[0])
            self.rng.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def predict(self, X_test):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.
        Args:
            X_test (np.ndarray): Test data points with shape (N, D)
        Returns:
            np.ndarray: Predictive mean of the test data points
            np.ndarray: Predictive variance of the test data points
        """
        # Basic shape check
        assert len(X_test.shape) == 2, "Incorrect shape for X_test."

        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Get features from the net
        theta = self.network.basis_funcs(torch.Tensor(X_)).data.numpy()

        # Marginalise predictions over hyperparameters of the BLR
        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])

        for i, m in enumerate(self.models):
            mu[i], var[i] = m.predict(theta)

        # See the algorithm runtime prediction paper by Hutter et al
        # for the derivation of the total variance
        m = np.mean(mu, axis=0)
        v = np.mean(mu ** 2 + var, axis=0) - m ** 2

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v

    def get_incumbent(self):
        """
        Returns the best observed point and its function value
        Returns:
            incumbent (np.ndarray): the best observed point
            incumbent_value (float): the function value at the best observed point
        """
        # Use self.X and self.y to compute the current best (lowest) objective value
        best_idx = np.argmin(self.y)
        incumbent = self.X[best_idx]
        incumbent_value = self.y[best_idx]

        if self.normalize_input:
            incumbent = zero_mean_unit_var_denormalization(incumbent, self.X_mean, self.X_std)

        if self.normalize_output:
            incumbent_value = zero_mean_unit_var_denormalization(incumbent_value, self.y_mean, self.y_std)

        return incumbent, incumbent_value

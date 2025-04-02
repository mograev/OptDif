"""
Model wrapper for temperature scaling.
Source: https://github.com/janschwedhelm/master-thesis/blob/main/src/temperature_scaling.py
"""

import torch
from torch import nn, optim


class ModelWithTemperature(nn.Module):
    """A thin decorator, which wraps a model with temperature scaling"""
    def __init__(self, model):
        """
        Initialize the model with temperature scaling.
        Args:
            model (nn.Module): The model to be wrapped.
        """
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        """
        Forward pass with temperature scaling.
        Args:
            input (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Scaled logits.
        """
        logits = self.model(input)

        # Convert list of logits to tensor
        if isinstance(logits, list):
            logits = torch.stack(logits, dim=0)
        
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits.
        Args:
            logits (torch.Tensor): Logits tensor.
        Returns:
            torch.Tensor: Scaled logits.
        """
        # Expand temperature to match the shape of logits
        # logits shape: [num_attributes, batch_size, num_classes]
        temperature = self.temperature.unsqueeze(-1).unsqueeze(-1)  # Shape: [1, 1, 1]
        temperature = temperature.expand(logits.size(0), logits.size(1), logits.size(2))  # Match logits shape
        return logits / temperature

    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using NLL on the validation set).
        Args:
            valid_loader (DataLoader): DataLoader for the validation set.
        Returns:
            ModelWithTemperature: The model with optimized temperature.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.model.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input, label = input.to(device), label.to(device)
                if self.attribute_idx:
                    logits = self.model(input)[self.attribute_idx]
                else:
                    logits = self.model(input)
                logits_list.append(logits)
                if self.attribute_idx:
                    labels_list.append(label[:, self.attribute_idx].long())
                else:
                    labels_list.append(label.long())
            logits = torch.cat(torch.tensor(logits_list)).cuda()
            labels = torch.cat(torch.tensor(labels_list)).cuda()

        # Calculate NLL before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f' % before_temperature_nll)

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f' % after_temperature_nll)

        return self
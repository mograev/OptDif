"""
Utility functions used in Stable Diffusion models.
Source: https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/diffusionmodules/util.py
"""

import torch
import importlib


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    Args:
        func (callable): The function to evaluate.
        inputs (list): List of input tensors.
        params (list): List of parameters.
        flag (bool): If True, use checkpointing.
    Returns:
        torch.Tensor: The output of the function.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    """
    A custom autograd function that allows for checkpointing of intermediate
    activations during the forward pass, reducing memory usage at the expense
    of extra compute in the backward pass.
    """
    @staticmethod
    def forward(ctx, run_function, length, *args):
        """
        Forward pass of the checkpoint function.
        Args:
            ctx (torch.autograd.Function): The context object.
            run_function (callable): The function to evaluate.
            length (int): The number of input tensors.
            *args: The input tensors and parameters.
        Returns:
            torch.Tensor: The output of the function.
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        """
        Backward pass of the checkpoint function.
        Args:
            ctx (torch.autograd.Function): The context object.
            *output_grads: The gradients of the output tensors.
        Returns:
            tuple: The gradients of the input tensors and parameters.
        """
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
    

def instantiate_from_config(config):
    """
    Instantiates an object from a configuration dictionary.
    Args:
        config (dict): The configuration dictionary.
    Returns:
        object: The instantiated object.
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """
    Imports a class or function from a string representation.
    Args:
        string (str): The string representation of the class or function.
        reload (bool): If True, reload the module.
    Returns:
        object: The imported class or function.
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)